# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Callable, Sequence, cast

from tianshou.policy import BasePolicy

from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.utils import FiniteEnvType, LogWriter

from .vessel import TrainingVessel
from .trainer import Trainer


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

    vessel = TrainingVessel(
        simulator_fn=simulator_fn,
        state_interpreter=state_interpreter,
        action_interpreter=action_interpreter,
        policy=policy,
        test_initial_states=initial_states,
        reward=cast(Reward, reward),  # ignore none
    )
    trainer = Trainer(
        finite_env_type=finite_env_type,
        concurrency=concurrency,
        loggers=logger,
    )
    trainer.test(vessel)
