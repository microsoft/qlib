# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import dataclasses
import random
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Sequence, cast

import numpy as np
import pandas as pd
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy
from torch.utils.data import Dataset
from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment, use_cuda
from utilsd.experiment import print_config
from utilsd.earlystop import EarlyStop, EarlyStopStatus
from utilsd.logging import print_log
from utilsd.config import RegistryConfig

from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.utils import DataQueue, EnvWrapper, FiniteEnvType, LogCollector, LogWriter, finite_env_factory
from qlib.log import get_module_logger
from qlib.typehint import Literal

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

    metrics: dict
    """Metrics of produced in train/val/test. Cleared on every new iteration of training.
    In fit, validation metrics will be prefixed with ``val/``."""

    current_iter: int
    """Current iteration (collect) of training."""

    def __init__(
        self, *,
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
                    if f'{typename}{retry}' not in res:
                        res[f'{typename}{retry}'] = callback
        return res

    def create_env_wrapper(self):
        return EnvWrapper(
            simulator_fn,
            state_interpreter,
            action_interpreter,
            seed_iterator,
            reward,
            logger=LogCollector(min_loglevel=min_loglevel),
        )

    def test(
        self,
        simulator_fn: Callable[[InitialStateType], Simulator],
        state_interpreter: StateInterpreter,
        action_interpreter: ActionInterpreter,
        policy: BasePolicy,
        initial_states: Sequence[InitialStateType],
    ) -> None:
        ...

    def fit(self, vessel: TrainingVesselBase):

        if self.checkpoint_dir is not None:
            _resume_path = self.checkpoint_dir / "resume.pth"
        else:
            _resume_path = Path("/tmp/resume.pth")

        def _resume():
            nonlocal best_state_dict, cur_epoch
            if _resume_path.exists():
                print_log(f"Resume from checkpoint: {_resume_path}", __name__)
                data = torch.load(_resume_path)
                logger.load_state_dict(data["logger"])
                val_logger.load_state_dict(data["val_logger"])
                earlystop.load_state_dict(data["earlystop"])
                policy.load_state_dict(data["policy"])
                best_state_dict = data["policy_best"]
                if hasattr(policy, "optim"):
                    policy.optim.load_state_dict(data["optim"])
                cur_epoch = data["epoch"]

        def _checkpoint():
            torch.save(
                {
                    "logger": logger.state_dict(),
                    "val_logger": val_logger.state_dict(),
                    "earlystop": earlystop.state_dict(),
                    "policy": policy.state_dict(),
                    "policy_best": best_state_dict,
                    "optim": policy.optim.state_dict()
                    if hasattr(policy, "optim")
                    else None,
                    "epoch": cur_epoch,
                },
                _resume_path,
            )
            print_log(f"Checkpoint saved to {_resume_path}", __name__)

        logger = Logger(
            episode_per_collect,
            log_interval=500,
            tb_prefix="train",
            count_global="step",
        )
        val_logger = Logger(len(val_dataset), log_interval=2000, tb_prefix="val")
        earlystop = EarlyStop(patience=earlystop_patience)
        cur_epoch = 0
        train_env = data_fn = best_state_dict = None

        _resume()

        try:
            if (
                self.checkpoint_dir is not None
                and self.preserve_intermediate_checkpoints
            ):
                torch.save(
                    policy.state_dict(),
                    self.checkpoint_dir / f"epoch_{cur_epoch:04d}.pth",
                )

            while cur_epoch < max_epoch:
                cur_epoch += 1
                if train_env is None:
                    train_env, data_fn = env_fn(logger, train_dataset, True)

                logger.reset(f"Train Epoch [{cur_epoch}/{max_epoch}] Episode")
                val_logger.reset(f"Val Epoch [{cur_epoch}/{max_epoch}] Episode")

                collector_res = self._train_epoch(
                    policy,
                    train_env,
                    buffer_size=buffer_size,
                    episode_per_collect=episode_per_collect,
                    batch_size=batch_size,
                    update_per_collect=update_per_collect,
                )
                logger.write_summary(collector_res)

                if self.checkpoint_dir is not None:
                    torch.save(policy.state_dict(), self.checkpoint_dir / "latest.pth")
                    if self.preserve_intermediate_checkpoints:
                        torch.save(
                            policy.state_dict(),
                            self.checkpoint_dir / f"epoch_{cur_epoch:04d}.pth",
                        )

                if cur_epoch == max_epoch or cur_epoch % val_every_n_collect == 0:
                    data_fn.cleanup()  # do this to save memory
                    train_env = data_fn = None

                    val_result, _ = self.evaluate(
                        policy, env_fn, val_dataset, val_logger
                    )
                    val_logger.global_step = logger.global_step  # sync two loggers
                    val_logger.write_summary()

                    es = earlystop.step(val_result)
                    if es == EarlyStopStatus.BEST:
                        best_state_dict = copy.deepcopy(policy.state_dict())
                        if self.checkpoint_dir is not None:
                            torch.save(
                                best_state_dict, self.checkpoint_dir / "best.pth"
                            )
                        pd.DataFrame.from_records(val_logger.logs).to_csv(
                            get_output_dir() / "metrics_val.csv", index=False
                        )
                    elif es == EarlyStopStatus.STOP:
                        break

                _checkpoint()

        finally:
            if data_fn is not None:
                data_fn.cleanup()

        if best_state_dict is not None:
            policy.load_state_dict(best_state_dict)

        return logger, val_logger

        test_logger = Logger(
            len(test_dataset), log_interval=2000, prefix="Test Episode", tb_prefix="test"
        )
        test_logger.global_step = train_logger.global_step
        _, test_result = trainer.evaluate(policy, env_fn, test_dataset, test_logger)
        test_logger.write_summary()
        test_result.to_csv(get_output_dir() / "metrics.csv", index=False)
        return test_result


    def _call_callback_hooks(
        self,
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        log.debug(f"{self.__class__.__name__}: calling callback hook: {hook_name}")
        # TODO: remove if block in v1.8
        if hook_name in ("on_init_start", "on_init_end"):
            # these `Callback` hooks are the only ones that do not take a lightning module.
            # we also don't profile bc profiler hasn't been set yet
            for callback in self.callbacks:
                fn = getattr(callback, hook_name)
                if callable(fn):
                    fn(self, *args, **kwargs)
            return

        pl_module = self.lightning_module
        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = hook_name

        for callback in self.callbacks:
            fn = getattr(callback, hook_name)
            if callable(fn):
                with self.profiler.profile(f"[Callback]{callback.state_key}.{hook_name}"):
                    fn(self, self.lightning_module, *args, **kwargs)

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name


    def _train_epoch(
        self,
        policy: BasePolicy,
        train_env: BaseVectorEnv,
        *,
        buffer_size: int,
        episode_per_collect: int,
        batch_size: int,
        update_per_collect: int,
    ) -> Dict[str, Any]:
        # 1 epoch = 1 collect
        collector = Collector(
            policy, train_env, VectorReplayBuffer(buffer_size, len(train_env))
        )
        policy.train()
        col_result = collector.collect(n_episode=episode_per_collect)
        update_result = policy.update(
            0, collector.buffer, batch_size=batch_size, repeat=update_per_collect
        )
        return {
            "collect/" + k: np.mean(v)
            for k, v in {**col_result, **update_result}.items()
        }

    def fit(
        self,

        simulator_fn: Callable[[InitialStateType], Simulator],
        state_interpreter: StateInterpreter,
        action_interpreter: ActionInterpreter,
        policy: BasePolicy,
        logger: LogWriter | list[LogWriter],
        reward: Reward
        initial_states: Sequence[InitialStateType],
        policy: BasePolicy,
        env_fn: Callable[
            [Logger, Dataset, bool], Tuple[BaseVectorEnv, DataConsumerFactory]
        ],
        train_dataset: Dataset,
        val_dataset: Dataset,
        *,
        max_epoch: int,
        update_per_collect: int,
        batch_size: int,
        episode_per_collect: int,
        buffer_size: int = 200000,
        earlystop_patience: int = 5,
        val_every_n_collect: int = 1,
    ) -> Tuple[Logger, Logger]:
        if self.checkpoint_dir is not None:
            _resume_path = self.checkpoint_dir / "resume.pth"
        else:
            _resume_path = Path("/tmp/resume.pth")

        def _resume():
            nonlocal best_state_dict, cur_epoch
            if _resume_path.exists():
                print_log(f"Resume from checkpoint: {_resume_path}", __name__)
                data = torch.load(_resume_path)
                logger.load_state_dict(data["logger"])
                val_logger.load_state_dict(data["val_logger"])
                earlystop.load_state_dict(data["earlystop"])
                policy.load_state_dict(data["policy"])
                best_state_dict = data["policy_best"]
                if hasattr(policy, "optim"):
                    policy.optim.load_state_dict(data["optim"])
                cur_epoch = data["epoch"]

        def _checkpoint():
            torch.save(
                {
                    "logger": logger.state_dict(),
                    "val_logger": val_logger.state_dict(),
                    "earlystop": earlystop.state_dict(),
                    "policy": policy.state_dict(),
                    "policy_best": best_state_dict,
                    "optim": policy.optim.state_dict()
                    if hasattr(policy, "optim")
                    else None,
                    "epoch": cur_epoch,
                },
                _resume_path,
            )
            print_log(f"Checkpoint saved to {_resume_path}", __name__)

        logger = Logger(
            episode_per_collect,
            log_interval=500,
            tb_prefix="train",
            count_global="step",
        )
        val_logger = Logger(len(val_dataset), log_interval=2000, tb_prefix="val")
        earlystop = EarlyStop(patience=earlystop_patience)
        cur_epoch = 0
        train_env = data_fn = best_state_dict = None

        _resume()

        try:
            if (
                self.checkpoint_dir is not None
                and self.preserve_intermediate_checkpoints
            ):
                torch.save(
                    policy.state_dict(),
                    self.checkpoint_dir / f"epoch_{cur_epoch:04d}.pth",
                )

            while cur_epoch < max_epoch:
                cur_epoch += 1
                if train_env is None:
                    train_env, data_fn = env_fn(logger, train_dataset, True)

                logger.reset(f"Train Epoch [{cur_epoch}/{max_epoch}] Episode")
                val_logger.reset(f"Val Epoch [{cur_epoch}/{max_epoch}] Episode")

                collector_res = self._train_epoch(
                    policy,
                    train_env,
                    buffer_size=buffer_size,
                    episode_per_collect=episode_per_collect,
                    batch_size=batch_size,
                    update_per_collect=update_per_collect,
                )
                logger.write_summary(collector_res)

                if self.checkpoint_dir is not None:
                    torch.save(policy.state_dict(), self.checkpoint_dir / "latest.pth")
                    if self.preserve_intermediate_checkpoints:
                        torch.save(
                            policy.state_dict(),
                            self.checkpoint_dir / f"epoch_{cur_epoch:04d}.pth",
                        )

                if cur_epoch == max_epoch or cur_epoch % val_every_n_collect == 0:
                    data_fn.cleanup()  # do this to save memory
                    train_env = data_fn = None

                    val_result, _ = self.evaluate(
                        policy, env_fn, val_dataset, val_logger
                    )
                    val_logger.global_step = logger.global_step  # sync two loggers
                    val_logger.write_summary()

                    es = earlystop.step(val_result)
                    if es == EarlyStopStatus.BEST:
                        best_state_dict = copy.deepcopy(policy.state_dict())
                        if self.checkpoint_dir is not None:
                            torch.save(
                                best_state_dict, self.checkpoint_dir / "best.pth"
                            )
                        pd.DataFrame.from_records(val_logger.logs).to_csv(
                            get_output_dir() / "metrics_val.csv", index=False
                        )
                    elif es == EarlyStopStatus.STOP:
                        break

                _checkpoint()

        finally:
            if data_fn is not None:
                data_fn.cleanup()

        if best_state_dict is not None:
            policy.load_state_dict(best_state_dict)

        return logger, val_logger

    def evaluate(
        self,
        policy: BasePolicy,
        env_fn: Callable[
            [Logger, Dataset, bool], Tuple[BaseVectorEnv, DataConsumerFactory]
        ],
        dataset: Dataset,
        logger: Optional[Logger] = None,
    ):
        if logger is None:
            logger = Logger(len(dataset))
        try:
            venv, data_fn = env_fn(logger, dataset, False)
            test_collector = Collector(policy, venv)
            policy.eval()
            test_collector.collect(n_step=int(1e18) * len(venv))
        except StopIteration:
            pass
        finally:
            data_fn.cleanup()

        return logger.summary()["reward"], pd.DataFrame.from_records(logger.logs)
