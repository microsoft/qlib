# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Callable, Sequence

from tianshou.data import Collector
from tianshou.policy import BasePolicy

from qlib.constant import INF
from qlib.log import get_module_logger
from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.utils.data_queue import DataQueue
from qlib.rl.utils.env_wrapper import EnvWrapper
from qlib.rl.utils.finite_env import FiniteEnvType, finite_env_cls
from qlib.rl.utils.logger import BasicLogger


_logger = get_module_logger(__name__)


def backtest(
    simulator_fn: Callable[[InitialStateType], Simulator],
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    seed_set: Sequence[InitialStateType],
    policy: BasePolicy,
    reward: Reward | None = None,
    finite_env_type: FiniteEnvType = 'subproc',
    concurrency: int = 2
):
    """Backtest with the parallelism provided by RL framework."""
    seed_iterator = DataQueue(seed_set)
    finite_venv = finite_env_cls(finite_env_type)

    with seed_iterator:
        vector_env = finite_venv(BasicLogger(), [
            lambda: EnvWrapper(simulator_fn, state_interpreter, action_interpreter, seed_iterator, reward)
            for _ in range(concurrency)
        ])

        try:
            test_collector = Collector(policy, vector_env)
        except StopIteration:
            pass

        policy.eval()
        _logger.info("All ready. Start backtest.", __name__)
        test_collector.collect(n_step=INF * len(vector_env))

    # logger.write_summary()

    # return pd.DataFrame.from_records(logger.logs), logger.history
