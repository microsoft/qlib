# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, Generic, Iterable, Sequence, TypeVar, cast

import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy

from qlib.constant import INF
from qlib.log import get_module_logger
from qlib.rl.interpreter import ActionInterpreter, ActType, ObsType, PolicyActType, StateInterpreter, StateType
from qlib.rl.reward import Reward
from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.utils import DataQueue
from qlib.rl.utils.finite_env import FiniteVectorEnv

if TYPE_CHECKING:
    from .trainer import Trainer


T = TypeVar("T")
_logger = get_module_logger(__name__)


class SeedIteratorNotAvailable(BaseException):
    pass


class TrainingVesselBase(Generic[InitialStateType, StateType, ActType, ObsType, PolicyActType]):
    """A ship that contains simulator, interpreter, and policy, will be sent to trainer.
    This class controls algorithm-related parts of training, while trainer is responsible for runtime part.

    The ship also defines the most important logic of the core training part,
    and (optionally) some callbacks to insert customized logics at specific events.
    """

    simulator_fn: Callable[[InitialStateType], Simulator[InitialStateType, StateType, ActType]]
    state_interpreter: StateInterpreter[StateType, ObsType]
    action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType]
    policy: BasePolicy
    reward: Reward
    trainer: Trainer

    def assign_trainer(self, trainer: Trainer) -> None:
        self.trainer = weakref.proxy(trainer)  # type: ignore

    def train_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        """Override this to create a seed iterator for training.
        If the iterable is a context manager, the whole training will be invoked in the with-block,
        and the iterator will be automatically closed after the training is done."""
        raise SeedIteratorNotAvailable("Seed iterator for training is not available.")

    def val_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        """Override this to create a seed iterator for validation."""
        raise SeedIteratorNotAvailable("Seed iterator for validation is not available.")

    def test_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        """Override this to create a seed iterator for testing."""
        raise SeedIteratorNotAvailable("Seed iterator for testing is not available.")

    def train(self, vector_env: BaseVectorEnv) -> Dict[str, Any]:
        """Implement this to train one iteration. In RL, one iteration usually refers to one collect."""
        raise NotImplementedError()

    def validate(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        """Implement this to validate the policy once."""
        raise NotImplementedError()

    def test(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        """Implement this to evaluate the policy on test environment once."""
        raise NotImplementedError()

    def log(self, name: str, value: Any) -> None:
        # FIXME: this is a workaround to make the log at least show somewhere.
        # Need a refactor in logger to formalize this.
        if isinstance(value, (np.ndarray, list)):
            value = np.mean(value)
        _logger.info(f"[Iter {self.trainer.current_iter + 1}] {name} = {value}")

    def log_dict(self, data: Dict[str, Any]) -> None:
        for name, value in data.items():
            self.log(name, value)

    def state_dict(self) -> Dict:
        """Return a checkpoint of current vessel state."""
        return {"policy": self.policy.state_dict()}

    def load_state_dict(self, state_dict: Dict) -> None:
        """Restore a checkpoint from a previously saved state dict."""
        self.policy.load_state_dict(state_dict["policy"])


class TrainingVessel(TrainingVesselBase):
    """The default implementation of training vessel.

    ``__init__`` accepts a sequence of initial states so that iterator can be created.
    ``train``, ``validate``, ``test`` each do one collect (and also update in train).
    By default, the train initial states will be repeated infinitely during training,
    and collector will control the number of episodes for each iteration.
    In validation and testing, the val / test initial states will be used exactly once.

    Extra hyper-parameters (only used in train) include:

    - ``buffer_size``: Size of replay buffer.
    - ``episode_per_iter``: Episodes per collect at training. Can be overridden by fast dev run.
    - ``update_kwargs``: Keyword arguments appearing in ``policy.update``.
      For example, ``dict(repeat=10, batch_size=64)``.
    """

    def __init__(
        self,
        *,
        simulator_fn: Callable[[InitialStateType], Simulator[InitialStateType, StateType, ActType]],
        state_interpreter: StateInterpreter[StateType, ObsType],
        action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType],
        policy: BasePolicy,
        reward: Reward,
        train_initial_states: Sequence[InitialStateType] | None = None,
        val_initial_states: Sequence[InitialStateType] | None = None,
        test_initial_states: Sequence[InitialStateType] | None = None,
        buffer_size: int = 20000,
        episode_per_iter: int = 1000,
        update_kwargs: Dict[str, Any] = cast(Dict[str, Any], None),
    ):
        self.simulator_fn = simulator_fn  # type: ignore
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter
        self.policy = policy
        self.reward = reward
        self.train_initial_states = train_initial_states
        self.val_initial_states = val_initial_states
        self.test_initial_states = test_initial_states
        self.buffer_size = buffer_size
        self.episode_per_iter = episode_per_iter
        self.update_kwargs = update_kwargs or {}

    def train_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        if self.train_initial_states is not None:
            _logger.info("Training initial states collection size: %d", len(self.train_initial_states))
            # Implement fast_dev_run here.
            train_initial_states = self._random_subset("train", self.train_initial_states, self.trainer.fast_dev_run)
            return DataQueue(train_initial_states, repeat=-1, shuffle=True)
        return super().train_seed_iterator()

    def val_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        if self.val_initial_states is not None:
            _logger.info("Validation initial states collection size: %d", len(self.val_initial_states))
            val_initial_states = self._random_subset("val", self.val_initial_states, self.trainer.fast_dev_run)
            return DataQueue(val_initial_states, repeat=1)
        return super().val_seed_iterator()

    def test_seed_iterator(self) -> ContextManager[Iterable[InitialStateType]] | Iterable[InitialStateType]:
        if self.test_initial_states is not None:
            _logger.info("Testing initial states collection size: %d", len(self.test_initial_states))
            test_initial_states = self._random_subset("test", self.test_initial_states, self.trainer.fast_dev_run)
            return DataQueue(test_initial_states, repeat=1)
        return super().test_seed_iterator()

    def train(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        """Create a collector and collects ``episode_per_iter`` episodes.
        Update the policy on the collected replay buffer.
        """
        self.policy.train()

        with vector_env.collector_guard():
            collector = Collector(self.policy, vector_env, VectorReplayBuffer(self.buffer_size, len(vector_env)))

            # Number of episodes collected in each training iteration can be overridden by fast dev run.
            if self.trainer.fast_dev_run is not None:
                episodes = self.trainer.fast_dev_run
            else:
                episodes = self.episode_per_iter

            col_result = collector.collect(n_episode=episodes)
            update_result = self.policy.update(sample_size=0, buffer=collector.buffer, **self.update_kwargs)
            res = {**col_result, **update_result}
            self.log_dict(res)
            return res

    def validate(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        self.policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(self.policy, vector_env)
            res = test_collector.collect(n_step=INF * len(vector_env))
            self.log_dict(res)
            return res

    def test(self, vector_env: FiniteVectorEnv) -> Dict[str, Any]:
        self.policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(self.policy, vector_env)
            res = test_collector.collect(n_step=INF * len(vector_env))
            self.log_dict(res)
            return res

    @staticmethod
    def _random_subset(name: str, collection: Sequence[T], size: int | None) -> Sequence[T]:
        if size is None:
            # Size = None -> original collection
            return collection
        order = np.random.permutation(len(collection))
        res = [collection[o] for o in order[:size]]
        _logger.info(
            "Fast running in development mode. Cut %s initial states from %d to %d.",
            name,
            len(collection),
            len(res),
        )
        return res
