

class BaseEnvWrapper:

    """
    # Base Env Wrapper for Reforcement Learning Framework

    class EnvWrapper(BaseEnvWrapper):
    """
    def __init__(self, sub_env, action_interpreter, state_interpreter):
        self.sub_env = sub_env
        self.action_interpreter = action_interpreter
        self.state_interpreter = state_interpreter

    def reset(self, **kwargs):
        self.upper_state = kwargs.get("upper_state", None)
        self.sub_env.reset()

    def step(self, action):
        sub_action = self.action_interpreter.interpret(action)
        sub_state = self.sub_env.step(sub_action)
        state = self.state_interpreter.interpret(sub_state)
        return state
        reurn self.
        if self.track:
            yield action
        yield from 

    def finished(self):
        return self.sub_env.finished()


        