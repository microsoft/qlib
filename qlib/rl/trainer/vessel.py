# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import weakref
from typing import Callable, Generic, Iterable, TYPE_CHECKING, Sequence, Any

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy

from qlib.constant import INF
from qlib.rl.interpreter import StateType, ActType, ObsType, PolicyActType
from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.utils import DataQueue, EnvWrapper, FiniteEnvType, LogCollector, LogWriter, finite_env_factory
from qlib.log import get_module_logger
from qlib.rl.utils.finite_env import FiniteVectorEnv
from qlib.typehint import Literal

if TYPE_CHECKING:
    from .trainer import Trainer


class TrainingVessel(Generic[InitialStateType, StateType, ActType, ObsType, PolicyActType]):
    """A ship that contains simulator, interpreter, and policy, will be sent to trainer.
    This class controls algorithm-related parts of training, while trainer is responible for runtime part.
    
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

    def train_seed_iterator(self) -> Iterable[InitialStateType]:
        """"""


    def val_seed_iterator(self) -> Iterable[InitialStateType]:
        return DataQueue(self.val_initial_states, repeat=1)

    def test_seed_iterator(self) -> Iterable[InitialStateType]:
        return DataQueue(self.test_initial_states, repeat=1)

    def train(self, vector_env: BaseVectorEnv) -> dict[str, Any]:
        collector = Collector(
            self.policy,
            vector_env,
            VectorReplayBuffer(self.buffer_size, len(vector_env))
        )
        self.policy.train()
        col_result = collector.collect(n_episode=self.episode_per_collect)
        update_result = self.policy.update(
            0, collector.buffer, batch_size=self.batch_size, repeat=self.update_per_collect
        )
        return {**col_result, **update_result}

    def validate(self, vector_env: FiniteVectorEnv) -> dict[str, Any]:
        self.policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(self.policy, vector_env)
            return test_collector.collect(n_step=INF * len(vector_env))

    def test(self, vector_env: FiniteVectorEnv) -> dict[str, Any]:
        self.policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(self.policy, vector_env)
            return test_collector.collect(n_step=INF * len(vector_env))


class DefaultTrainingVessel(TrainingVessel):
    """The default implementation of training vessel.
    
    ``__init__`` accepts a sequence of initial states so that iterator can be created.
    ``train``, ``validate``, ``test`` each do one collect (and also update in train).

    Extra hyper-parameters (only used in train) include:

    - ``buffer_size``: Size of replay buffer.
    - ``episode_per_collect``: Episodes per collect at training.
    - ``update_per_collect``: Number of updates in ``self.policy.update`` after each collect.
    - ``batch_size``: Batch size in ``self.policy.update`` after each collect.
    """

    def __init__(self, *,
        simulator_fn: Callable[[InitialStateType], Simulator[InitialStateType, StateType, ActType]],
        state_interpreter: StateInterpreter[StateType, ObsType],
        action_interpreter: ActionInterpreter[StateType, PolicyActType, ActType],
        policy: BasePolicy,
        reward: Reward,
        buffer_size: int,
        episode_per_collect: int,
        update_per_collect: int,
        batch_size: int
    ):
        ...


    train_initial_states: Sequence[InitialStateType]
    val_initial_states: Sequence[InitialStateType]
    test_initial_states: Sequence[InitialStateType]

    buffer_size: int
    episode_per_collect: int
    update_per_collect: int
    batch_size: int


    def train_seed_iterator(self) -> Iterable[InitialStateType]:
        return DataQueue(self.train_initial_states, repeat=-1, shuffle=True)

    def val_seed_iterator(self) -> Iterable[InitialStateType]:
        return DataQueue(self.val_initial_states, repeat=1)

    def test_seed_iterator(self) -> Iterable[InitialStateType]:
        return DataQueue(self.test_initial_states, repeat=1)

    def train_one_collect(
        self,
        vector_env: BaseVectorEnv,
    ) -> dict[str, Any]:
        collector = Collector(
            self.policy,
            vector_env,
            VectorReplayBuffer(self.buffer_size, len(vector_env))
        )
        self.policy.train()
        col_result = collector.collect(n_episode=self.episode_per_collect)
        update_result = self.policy.update(
            0, collector.buffer, batch_size=self.batch_size, repeat=self.update_per_collect
        )
        return {**col_result, **update_result}

    def validate(self, vector_env: FiniteVectorEnv) -> dict[str, Any]:
        self.policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(self.policy, vector_env)
            return test_collector.collect(n_step=INF * len(vector_env))

    def test(self, vector_env: FiniteVectorEnv) -> dict[str, Any]:
        self.policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(self.policy, vector_env)
            return test_collector.collect(n_step=INF * len(vector_env))
