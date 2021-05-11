# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .interpreter import StateInterpreter, ActionInterpreter

from ..contrib.backtest.executor import BaseExecutor


class BaseRLEnv:
    def reset(self, **kwargs):
        raise NotImplementedError("reset is not implemented!")

    def step(self, action):
        """
        step method of rl env
        Parameters
        ----------
        action :
            action from rl policy

        Returns
        -------
        env state to rl policy
        """
        raise NotImplementedError("step is not implemented!")


class QlibRLEnv:
    """qlib-based RL env"""

    def __init__(
        self,
        executor: BaseExecutor,
    ):
        """
        Parameters
        ----------
        executor : BaseExecutor
            qlib multi-level/single-level executor, which can be regarded as gamecore in RL
        """
        self.executor = executor

    def reset(self, **kwargs):
        self.executor.reset(**kwargs)


class QlibIntRLEnv(QlibRLEnv):
    """(Qlib)-based RL (Env) with (Interpreter)"""

    def __init__(
        self,
        executor: BaseExecutor,
        state_interpreter: StateInterpreter,
        action_interpreter: ActionInterpreter,
        state_interpret_kwargs: dict = {},
        action_interpret_kwargs: dict = {},
    ):
        """

        Parameters
        ----------
        state_interpreter : StateInterpreter
            interpretor that interprets the qlib execute result into rl env state.
        action_interpreter : ActionInterpreter
            interpretor that interprets the rl agent action into qlib order list
        state_interpret_kwargs : dict, optional
            arguments may be used in `state_interpreter.interpret`, by default {}
            such as the following arguments:
                - trade exchange : Exchange
                    Exchange that can provide market info
        action_interpret_kwargs: dict, optional
            arguments may be used in `action_interpreter.interpret`, by default {}
            such as the following arguments:
                - trade_order_list : List[Order]
                    If the strategy is used to split order, it presents the trade order pool.
        """
        super(QlibIntRLEnv, self).__init__(executor=executor)
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter
        self.state_interpret_kwargs = state_interpret_kwargs
        self.action_interpret_kwargs = action_interpret_kwargs

    def step(self, action):
        """
        step method of rl env, it run as following step:
            - Use `action_interpreter.interpret` method to interpret the agent action into order list
            - Execute the order list with qlib executor, and get the executed result
            - Use `state_interpreter.interpret` method to interpret the executed result into env state

        Parameters
        ----------
        action :
            action from rl policy

        Returns
        -------
        env state to rl rl policy
        """
        _interpret_action = self.action_interpreter.interpret(action=action, **self.state_interpret_kwargs)
        _execute_result = self.executor.execute(_interpret_action)
        _interpret_state = self.state_interpreter.interpret(
            execute_result=_execute_result, **self.action_interpret_kwargs
        )
        return _interpret_state
