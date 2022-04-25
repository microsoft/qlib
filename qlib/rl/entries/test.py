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
from qlib.rl.utils import DataQueue, EnvWrapper, FiniteEnvType, LogCollector, LogWriter, finite_env_factory


_logger = get_module_logger(__name__)


def backtest(
    simulator_fn: Callable[[InitialStateType], Simulator],
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    initial_states: Sequence[InitialStateType],
    policy: BasePolicy,
    logger: LogWriter | list[LogWriter],
    reward: Reward | None = None,
    finite_env_type: FiniteEnvType = "subproc",
    concurrency: int = 2
) -> None:
    """Backtest with the parallelism provided by RL framework.

    Parameters
    ----------
    simulator_fn
        Callable receiving initial seed, returning a simulator.
    state_interpreter
        Interprets the state of simulators.
    action_interpreter
        Interprets the policy actions.
    initial_states
        Initial states to iterate over. Every state will be run exactly once.
    policy
        Policy to test against.
    logger
        Logger to record the backtest results. Logger must be present because
        without logger, all information will be lost.
    reward
        Optional reward function. For backtest, this is for testing the rewards
        and logging them only.
    finite_env_type
        Type of finite env implementation.
    concurrency
        Parallel workers.
    """

    # To save bandwidth
    min_loglevel = min(l.loglevel for l in logger) if isinstance(logger, list) else logger.loglevel

    with DataQueue(initial_states) as seed_iterator:
        vector_env = finite_env_factory(
            lambda: EnvWrapper(
                simulator_fn,
                state_interpreter,
                action_interpreter,
                seed_iterator,
                reward,
                logger=LogCollector(min_loglevel=min_loglevel)
            ),
            finite_env_type,
            concurrency,
            logger
        )

        policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(policy, vector_env)
            _logger.info("All ready. Start backtest.")
            test_collector.collect(n_step=INF * len(vector_env))
