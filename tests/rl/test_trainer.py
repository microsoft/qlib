import random

import torch
import torch.nn as nn
from gym import spaces

from qlib.config import C
from qlib.log import set_log_with_config
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.simulator import Simulator
from qlib.rl.reward import Reward
from qlib.rl.trainer import Trainer, TrainingVessel



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
        'acc': spaces.Discrete(200),
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
    def __init__(self, out_features=1, return_state=False):
        super().__init__()
        self.fc = nn.Linear(32, out_features)
        self.return_state = return_state

    def forward(self, obs, state=None, **kwargs):
        res = self.fc(torch.randn(obs['acc'].size(0), 32))
        if self.return_state:
            return res, state
        else:
            return res


def test_trainer():
    set_log_with_config(C.logging_config)
    trainer = Trainer(max_iters=10)
    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        action_interpreter=NoopActionInterpreter(),
        policy=PolicyNet(),
        train_initial_states=list(range(100)),
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        update_kwargs=dict(repeat=10, batch_size=64),
    )
    trainer.fit(vessel)


test_trainer()