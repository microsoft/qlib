from typing import Callable, Any, Iterator, Optional

import numpy as np
import gym

from .simulator import Simulator
from .interpreter import StateInterpreter, ActionInterpreter

InitialStateType = Any


def fill_invalid(obj):
    if isinstance(obj, (int, float, bool)):
        return fill_invalid(np.array(obj))
    if hasattr(obj, "dtype"):
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.floating):
                return np.full_like(obj, np.nan)
            return np.full_like(obj, np.iinfo(obj.dtype).max)
            return obj
        # dealing with corner cases that numpy number is not supported by tianshou's sharray
        return fill_invalid(np.array(obj))
    elif isinstance(obj, dict):
        return {k: fill_invalid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fill_invalid(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([fill_invalid(v) for v in obj])
    raise ValueError(f"Unsupported value to fill with invalid: {obj}")


def isinvalid(arr):
    if hasattr(arr, "dtype"):
        if np.issubdtype(arr.dtype, np.floating):
            return np.isnan(arr).all()
        return (np.iinfo(arr.dtype).max == arr).all()
    if isinstance(arr, dict):
        return all([isinvalid(o) for o in arr.values()])
    if isinstance(arr, (list, tuple)):
        return all([isinvalid(o) for o in arr])
    if isinstance(arr, (int, float, bool, np.number)):
        return isinvalid(np.array(arr))
    return True


def generate_nan_observation(obs_space: gym.Space) -> Any:
    # We assume that obs is complex and there must be something like float
    # otherwise this logic will not be accurate
    sample = obs_space.sample()
    sample = fill_invalid(sample)
    return sample


def check_nan_observation(obs: Any) -> bool:
    return isinvalid(obs)


class SingleEnvWrapper(gym.Env):
    """Qlib-based RL environment.
    This is a wrapper of componenets, including simulator, state-interpreter, action-interpreter, reward.
    """

    def __init__(
        self,
        simulator_fn: Callable[[InitialStateType], Simulator],
        state_interpreter: StateInterpreter,
        action_interpreter: ActionInterpreter,
        initial_state_queue: Optional[Iterator[InitialStateType]],
    ):
        self.simulator_fn = simulator_fn
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter
        self.initial_state_queue = initial_state_queue

    @property
    def action_space(self):
        return self.action_interpreter.action_space

    @property
    def observation_space(self):
        return self.state_interpreter.observation_space

    def reset(self, **kwargs):
        try:
            cur_order = next(self.initial_state_queue)
            self.simulator = self.simulator_fn(cur_order)
        except StopIteration:
            self.dataloader = None
            return generate_nan_observation(self.observation_space)


class InitialStateDistributor:
    ...
