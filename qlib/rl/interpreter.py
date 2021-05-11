# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class BaseInterpreter:
    @staticmethod
    def interpret(**kwargs):
        raise NotImplementedError("interpret is not implemented!")


class ActionInterpreter(BaseInterpreter):
    @staticmethod
    def interpret(action, **kwargs):
        raise NotImplementedError("interpret is not implemented!")


class StateInterpreter(BaseInterpreter):
    @staticmethod
    def interpret(execute_result, **kwargs):
        raise NotImplementedError("interpret is not implemented!")
