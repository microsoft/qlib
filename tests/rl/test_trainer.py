import random

import torch.nn as nn
from gym import spaces

from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.simulator import Simulator
from qlib.rl.reward import Reward



class ZeroSimulator(Simulator):

    def step(self, action):
        self.action = action
        self.correct = action == 0

    def get_state(self):
        return {
            'acc': self.correct * 100,
            'action': self.action,
        }

    def done(self) -> bool:
        return random.choice([False, True])


class NoopStateInterpreter(StateInterpreter):
    observation_space = spaces.Dict({
        'acc': spaces.Box(0, 100),
        'action': spaces.Discrete(2),
    })

    def interpret(self, simulator_state):
        return simulator_state


class NoopActionInterpreter(ActionInterpreter):
    action_space = spaces.Discrete(2)

    def interpret(self, simulator_state, action):
        return action


class AccReward(Reward):
    def reward(self, simulator_state):
        return simulator_state['acc'] / 100


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 1)