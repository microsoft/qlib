import os
import random
import sys
from pathlib import Path

import pytest

import torch
import torch.nn as nn
from gym import spaces
from tianshou.policy import PPOPolicy

from qlib.config import C
from qlib.log import set_log_with_config
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.simulator import Simulator
from qlib.rl.reward import Reward
from qlib.rl.trainer import Trainer, TrainingVessel, EarlyStopping, Checkpoint

pytestmark = pytest.mark.skipif(sys.version_info < (3, 8), reason="Pickle styled data only supports Python >= 3.8")


class ZeroSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        self.action = self.correct = 0

    def step(self, action):
        self.action = action
        self.correct = action == 0
        self._done = random.choice([False, True])
        if self._done:
            self.env.logger.add_scalar("acc", self.correct * 100)

    def get_state(self):
        return {
            "acc": self.correct * 100,
            "action": self.action,
        }

    def done(self) -> bool:
        return self._done


class NoopStateInterpreter(StateInterpreter):
    observation_space = spaces.Dict(
        {
            "acc": spaces.Discrete(200),
            "action": spaces.Discrete(2),
        }
    )

    def interpret(self, simulator_state):
        return simulator_state


class NoopActionInterpreter(ActionInterpreter):
    action_space = spaces.Discrete(2)

    def interpret(self, simulator_state, action):
        return action


class AccReward(Reward):
    def reward(self, simulator_state):
        if self.env.status["done"]:
            return simulator_state["acc"] / 100
        return 0.0


class PolicyNet(nn.Module):
    def __init__(self, out_features=1, return_state=False):
        super().__init__()
        self.fc = nn.Linear(32, out_features)
        self.return_state = return_state

    def forward(self, obs, state=None, **kwargs):
        res = self.fc(torch.randn(obs["acc"].shape[0], 32))
        if self.return_state:
            return nn.functional.softmax(res, dim=-1), state
        else:
            return res


def _ppo_policy():
    actor = PolicyNet(2, True)
    critic = PolicyNet()
    policy = PPOPolicy(
        actor,
        critic,
        torch.optim.Adam(tuple(actor.parameters()) + tuple(critic.parameters())),
        torch.distributions.Categorical,
        action_space=NoopActionInterpreter().action_space,
    )
    return policy


def test_trainer():
    set_log_with_config(C.logging_config)
    trainer = Trainer(max_iters=10, finite_env_type="subproc")
    policy = _ppo_policy()

    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        action_interpreter=NoopActionInterpreter(),
        policy=policy,
        train_initial_states=list(range(100)),
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        episode_per_iter=500,
        update_kwargs=dict(repeat=10, batch_size=64),
    )
    trainer.fit(vessel)
    assert trainer.current_iter == 10
    assert trainer.current_episode == 5000
    assert abs(trainer.metrics["acc"] - trainer.metrics["reward"] * 100) < 1e-4
    assert trainer.metrics["acc"] > 80
    trainer.test(vessel)
    assert trainer.metrics["acc"] > 60


def test_trainer_fast_dev_run():
    set_log_with_config(C.logging_config)
    trainer = Trainer(max_iters=2, fast_dev_run=2, finite_env_type="shmem")
    policy = _ppo_policy()

    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        action_interpreter=NoopActionInterpreter(),
        policy=policy,
        train_initial_states=list(range(100)),
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        episode_per_iter=500,
        update_kwargs=dict(repeat=10, batch_size=64),
    )
    trainer.fit(vessel)
    assert trainer.current_episode == 4


def test_trainer_earlystop():
    # TODO this is just sanity check.
    # need to see the logs to check whether it works.
    set_log_with_config(C.logging_config)
    trainer = Trainer(
        max_iters=10,
        val_every_n_iters=1,
        finite_env_type="dummy",
        callbacks=[EarlyStopping("val/reward", restore_best_weights=True)],
    )
    policy = _ppo_policy()

    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        action_interpreter=NoopActionInterpreter(),
        policy=policy,
        train_initial_states=list(range(100)),
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        episode_per_iter=500,
        update_kwargs=dict(repeat=10, batch_size=64),
    )
    trainer.fit(vessel)
    assert trainer.metrics["val/acc"] > 30
    assert trainer.current_iter == 2  # second iteration


def test_trainer_checkpoint():
    set_log_with_config(C.logging_config)
    output_dir = Path(__file__).parent / ".output"
    trainer = Trainer(max_iters=2, finite_env_type="dummy", callbacks=[Checkpoint(output_dir, every_n_iters=1)])
    policy = _ppo_policy()

    vessel = TrainingVessel(
        simulator_fn=lambda init: ZeroSimulator(init),
        state_interpreter=NoopStateInterpreter(),
        action_interpreter=NoopActionInterpreter(),
        policy=policy,
        train_initial_states=list(range(100)),
        val_initial_states=list(range(10)),
        test_initial_states=list(range(10)),
        reward=AccReward(),
        episode_per_iter=100,
        update_kwargs=dict(repeat=10, batch_size=64),
    )
    trainer.fit(vessel)

    assert (output_dir / "001.pth").exists()
    assert (output_dir / "002.pth").exists()
    assert os.readlink(output_dir / "latest.pth") == str(output_dir / "002.pth")

    trainer.load_state_dict(torch.load(output_dir / "001.pth", weights_only=False))
    assert trainer.current_iter == 1
    assert trainer.current_episode == 100

    # Reload the checkpoint at first iteration
    trainer.fit(vessel, ckpt_path=output_dir / "001.pth")
