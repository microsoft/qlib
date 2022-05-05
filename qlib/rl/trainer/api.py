# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
from typing import Callable, Sequence

from tianshou.data import Collector
from tianshou.policy import BasePolicy

from qlib.constant import INF
from qlib.log import get_module_logger
from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.utils import DataQueue, EnvWrapper, FiniteEnvType, LogCollector, LogWriter, vectorize_env


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
    concurrency: int = 2,
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
    min_loglevel = min(lg.loglevel for lg in logger) if isinstance(logger, list) else logger.loglevel

    def env_factory():
        # FIXME: state_interpreter and action_interpreter are stateful (having a weakref of env),
        # and could be thread unsafe.
        # I'm not sure whether it's a design flaw.
        # I'll rethink about this when designing the trainer.

        if finite_env_type == "dummy":
            # We could only experience the "threading-unsafe" problem in dummy.
            state = copy.deepcopy(state_interpreter)
            action = copy.deepcopy(action_interpreter)
            rew = copy.deepcopy(reward)
        else:
            state, action, rew = state_interpreter, action_interpreter, reward

        return EnvWrapper(
            simulator_fn,
            state,
            action,
            seed_iterator,
            rew,
            logger=LogCollector(min_loglevel=min_loglevel),
        )

    with DataQueue(initial_states) as seed_iterator:
        vector_env = vectorize_env(
            env_factory,
            finite_env_type,
            concurrency,
            logger,
        )

        policy.eval()

        with vector_env.collector_guard():
            test_collector = Collector(policy, vector_env)
            _logger.info("All ready. Start backtest.")
            test_collector.collect(n_step=INF * len(vector_env))
