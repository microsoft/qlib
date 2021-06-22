# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class BaseInterpreter:
    """Base Interpreter"""

    def interpret(self, **kwargs):
        raise NotImplementedError("interpret is not implemented!")


class ActionInterpreter(BaseInterpreter):
    """Action Interpreter that interpret rl agent action into qlib orders"""

    def interpret(self, action, **kwargs):
        """interpret method

        Parameters
        ----------
        action :
            rl agent action

        Returns
        -------
        qlib orders

        """

        raise NotImplementedError("interpret is not implemented!")


class StateInterpreter(BaseInterpreter):
    """State Interpreter that interpret execution result of qlib executor into rl env state"""

    def interpret(self, execute_result, **kwargs):
        """interpret method

        Parameters
        ----------
        execute_result :
            qlib execution result

        Returns
        ----------
        rl env state
        """
        raise NotImplementedError("interpret is not implemented!")
