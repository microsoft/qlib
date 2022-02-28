from typing import Callable, Any, Iterator, Optional

import numpy as np
import gym

from qlib.rl.simulator import Simulator, InitialStateType
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter

from .finite_env import generate_nan_observation



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
        info_collector: Optional[InfoCollector] = None
    ):
        self.simulator_fn = simulator_fn
        self.state_interpreter = state_interpreter
        self.action_interpreter = action_interpreter
        self.initial_state_queue = initial_state_queue
        self.info_collector = info_collector

    @property
    def action_space(self):
        return self.action_interpreter.action_space

    @property
    def observation_space(self):
        return self.state_interpreter.observation_space

    def reset(self, **kwargs):
        # Try to get a state from state queue, and init the simulator with this state.
        # If the queue is exhausted, generate an invalid (nan) observation
        try:
            cur_order = next(self.initial_state_queue)
            self.simulator = self.simulator_fn(cur_order)
            return self.simulator.get_state()
        except StopIteration:
            self.initial_state_queue = None
            return generate_nan_observation(self.observation_space)

    def step(self, action, **kwargs):
        if self.initial_state_queue is None:
            raise RuntimeError('State queue is already exhausted, but the environment is still receiving action.')

        # Action is what we have got from policy
        action = self.action_interpreter(action, self.simulator.get_state())
        self.simulator.step(action)

        sim_state = self.simulator.get_state()

        obs = self.state_interpreter(sim_state)
        rew = self.reward_fn(sim_state)
        done = self.simulator.done()
        info = self.info_collector(sim_state)

        return obs, rew, done, info
