# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Iterable, cast

import torch

from qlib.rl.simulator import InitialStateType
from qlib.rl.utils import EnvWrapper, FiniteEnvType, LogCollector, LogWriter, vectorize_env, LogLevel
from qlib.log import get_module_logger

from .callbacks import Callback
from .vessel import TrainingVesselBase

_logger = get_module_logger(__name__)


class Trainer:
    """
    Utility to train a policy on a particular task.

    Different from traditional DL trainer, the iteration of this trainer is "collect",
    rather than "epoch", or "mini-batch".
    In each collect, :class:`Collector` collects a number of policy-env interactions, and accumulates
    them into a replay buffer. This buffer is used as the "data" to train the policy.
    At the end of each collect, the policy is *updated* several times.

    The API has some resemblence with `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`__,
    but it's essentially different because this trainer is built for RL applications, and thus
    most configurations are under RL context.
    We are still looking for ways to incorporate existing trainer libraries, because it looks like
    big efforts to build a trainer as powerful as those libraries, and also, that's not our primary goal.

    It's essentially different
    `tianshou's built-in trainers <https://tianshou.readthedocs.io/en/master/api/tianshou.trainer.html>`__,
    as it's far much more complicated than that.

    Parameters
    ----------
    max_iters
        Maximum iterations before stopping.
    val_every_n_iters
        Perform validation every n iterations (i.e., training collects).
    logger
        Logger to record the backtest results. Logger must be present because
        without logger, all information will be lost.
    finite_env_type
        Type of finite env implementation.
    concurrency
        Parallel workers.
    fast_dev_run
        Create a subset for debugging.
        How this is implemented depends on the implementation of training vessel.
        For :class:`~qlib.rl.vessel.TrainingVessel`, if greater than zero,
        a random subset sized ``fast_dev_run`` will be used
        instead of ``train_initial_states`` and ``val_initial_states``.
    """

    should_stop: bool
    """Set to stop the training."""

    # TODO: make metrics appear here

    metrics: dict
    """Metrics of produced in train/val/test. Cleared on every new iteration of training.
    In fit, validation metrics will be prefixed with ``val/``."""

    current_iter: int
    """Current iteration (collect) of training."""

    def __init__(
        self,
        *,
        max_iters: int | None = None,
        val_every_n_iters: int | None = None,
        logger: LogWriter | list[LogWriter] | None = None,
        callbacks: list[Callback] | None = None,
        finite_env_type: FiniteEnvType = "subproc",
        concurrency: int = 2,
        fast_dev_run: int = 0,
    ):
        self.max_iters = max_iters
        self.val_every_n_iters = val_every_n_iters

        if isinstance(logger, list):
            self.logger: list[LogWriter] = logger
        elif isinstance(logger, LogWriter):
            self.logger: list[LogWriter] = [logger]
        else:
            self.logger: list[LogWriter] = []

        self.callbacks: list[Callback] = callbacks if callbacks is not None else []
        self.finite_env_type = finite_env_type
        self.concurrency = concurrency
        self.fast_dev_run = fast_dev_run

        self.vessel: TrainingVesselBase = cast(TrainingVesselBase, None)

    def initialize(self):
        self.should_stop = False
        self.current_iter = 0

    def initialize_iter(self):
        self.metrics = {}

    def state_dict(self) -> dict:
        """Putting every states of current training into a dict, at best effort.

        It doesn't try to handle all the possible kinds of states in the middle of one training collect.
        For most cases at the end of each iteration, things should be usually correct.
        """
        return {
            "vessel": self.vessel.state_dict(),
            "callbacks": {name: callback.state_dict() for name, callback in self.named_callbacks().items()},
            "should_stop": self.should_stop,
            "current_iter": self.current_iter,
            "metrics": self.metrics,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load all states into current trainer."""
        self.vessel.load_state_dict(state_dict["vessel"])
        for name, callback in self.named_callbacks().items():
            callback.load_state_dict(state_dict["callbacks"][name])
        self.should_stop = state_dict["should_stop"]
        self.current_iter = state_dict["current_iter"]
        self.metrics = state_dict["metrics"]

    def named_callbacks(self) -> dict[str, Callback]:
        """Retrieve a collection of callbacks where each one has a name.
        Useful when saving checkpoints.
        """
        res = {}
        for callback in self.callbacks:
            typename = type(callback).__name__.lower()
            if typename not in res:
                res[typename] = callback
            else:
                # names are auto-labelled as earlystop1, earlystop2, ...
                for retry in range(1, 1000):
                    if f"{typename}{retry}" not in res:
                        res[f"{typename}{retry}"] = callback
        return res

    def fit(self, vessel: TrainingVesselBase, ckpt_path: Path | None = None) -> None:
        """Train the RL policy upon the defined simulator.
        
        Parameters
        ----------
        vessel
            A bundle of all elements used in training.
        ckpt_path
            Load a pre-trained / paused training checkpoint.
        """
        vessel.assign_trainer(self)

        if ckpt_path is not None:
            _logger.info("Resuming states from %s", str(ckpt_path))
            self.load_state_dict(torch.load(ckpt_path))
        else:
            self.initialize()

        self._call_callback_hooks("on_fit_start")

        while self.current_iter < self.max_iters:
            self._call_callback_hooks("on_train_start")

            with _wrap_context(vessel.train_seed_iterator()) as iterator:
                vector_env = self.venv_from_iterator(iterator)
                self.vessel.train(vector_env)

            self._call_callback_hooks("on_train_end")

            if (self.current_iter + 1) % self.val_every_n_iters == 0:
                self._call_callback_hooks("on_validate_start")
                with vessel.val_seed_iterator() as iterator:
                    vector_env = self.venv_from_iterator(vessel.val_seed_iterator())
                    self.vessel.validate(vector_env)

                self._call_callback_hooks("on_validate_end")

            self.current_iter += 1

            if self.should_stop:
                break

        self._call_callback_hooks("on_fit_end")

    def test(self, vessel: TrainingVesselBase) -> None:
        """Test the RL policy against the simulator.

        The simulator will be fed with data generated in ``test_seed_iterator``.

        Parameters
        ----------
        vessel
            A bundle of all related elements.
        """
        vessel.assign_trainer(self)

        self._call_callback_hooks("on_test_start")
        with _wrap_context(vessel.test_seed_iterator()) as iterator:
            vector_env = self.venv_from_iterator(iterator)
            self.vessel.test(vector_env)
        self._call_callback_hooks("on_test_end")

    def venv_from_iterator(self, iterator: Iterable[InitialStateType]) -> None:
        """Create a vectorized environment from iterator and the training vessel."""

        if not self.logger:
            min_loglevel = LogLevel.PERIODIC
        else:
            # To save bandwidth
            min_loglevel = min(lg.loglevel for lg in self.logger)

        def env_factory():
            # FIXME: state_interpreter and action_interpreter are stateful (having a weakref of env),
            # and could be thread unsafe.
            # I'm not sure whether it's a design flaw.
            # I'll rethink about this when designing the trainer.

            if self.finite_env_type == "dummy":
                # We could only experience the "threading-unsafe" problem in dummy.
                state = copy.deepcopy(self.vessel.state_interpreter)
                action = copy.deepcopy(self.vessel.action_interpreter)
                rew = copy.deepcopy(self.vessel.reward)
            else:
                state = self.vessel.state_interpreter
                action = self.vessel.action_interpreter
                rew = self.vessel.reward

            return EnvWrapper(
                self.vessel.simulator_fn,
                state,
                action,
                iterator,
                rew,
                logger=LogCollector(min_loglevel=min_loglevel),
            )

        return vectorize_env(
            env_factory,
            self.finite_env_type,
            self.concurrency,
            self.logger,
        )

    def _call_callback_hooks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, hook_name)
            fn(self, self.vessel, *args, **kwargs)


@contextmanager
def _wrap_context(obj):
    """Make any object a (possibly dummy) context manager."""

    if isinstance(obj, AbstractContextManager):
        # obj has __enter__ and __exit__
        with obj as ctx:
            yield ctx
    else:
        yield obj
