
class BaseInterpreter:
    @staticmethod
    def interpret(**kwargs):
        raise NotImplementedError("interpret is not implemented!")

class ActionInterpreter:
    @staticmethod
    def interpret(action, **kwargs):
        return action

class StateInterpreter:
    @staticmethod
    def interpret(state, **kwargs):
        return state