# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from random import randint, choice
from pathlib import Path
import logging

import re
from typing import Any, Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces
from tianshou.data import Collector, Batch
from tianshou.policy import BasePolicy

from qlib.log import set_log_with_config
from qlib.config import C
from qlib.constant import INF
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.simulator import Simulator
from qlib.rl.utils.data_queue import DataQueue
from qlib.rl.utils.env_wrapper import InfoDict, EnvWrapper
from qlib.rl.utils.log import LogLevel, LogCollector, CsvWriter, ConsoleWriter
from qlib.rl.utils.finite_env import vectorize_env


class SimpleEnv(gym.Env[int, int]):
    def __init__(self) -> None:
        self.logger = LogCollector()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *args: Any, **kwargs: Any) -> int:
        self.step_count = 0
        return 0

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        self.logger.reset()

        self.logger.add_scalar("reward", 42.0)

        self.logger.add_scalar("a", randint(1, 10))
        self.logger.add_array("b", pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

        if self.step_count >= 3:
            done = choice([False, True])
        else:
            done = False

        if 2 <= self.step_count <= 3:
            self.logger.add_scalar("c", randint(11, 20))

        self.step_count += 1

        return 1, 42.0, done, InfoDict(log=self.logger.logs(), aux_info={})

    def render(self, mode: str = "human") -> None:
        pass


class AnyPolicy(BasePolicy):
    def forward(self, batch, state=None):
        return Batch(act=np.stack([1] * len(batch)))

    def learn(self, batch):
        pass


def test_simple_env_logger(caplog):
    set_log_with_config(C.logging_config)
    # In order for caplog to capture log messages, we configure it here:
    # allow logs from the qlib logger to be passed to the parent logger.
    C.logging_config["loggers"]["qlib"]["propagate"] = True
    logging.config.dictConfig(C.logging_config)
    for venv_cls_name in ["dummy", "shmem", "subproc"]:
        writer = ConsoleWriter()
        csv_writer = CsvWriter(Path(__file__).parent / ".output")
        venv = vectorize_env(lambda: SimpleEnv(), venv_cls_name, 4, [writer, csv_writer])
        with venv.collector_guard():
            collector = Collector(AnyPolicy(), venv)
            collector.collect(n_episode=30)

        output_file = pd.read_csv(Path(__file__).parent / ".output" / "result.csv")
        assert output_file.columns.tolist() == ["reward", "a", "c"]
        assert len(output_file) >= 30
    line_counter = 0
    for line in caplog.text.splitlines():
        line = line.strip()
        if line:
            line_counter += 1
            assert re.match(r".*reward .* {2}a .* \(([456])\.\d+\) {2}c .* \((14|15|16)\.\d+\)", line)
    assert line_counter >= 3


class SimpleSimulator(Simulator[int, float, float]):
    def __init__(self, initial: int, **kwargs: Any) -> None:
        super(SimpleSimulator, self).__init__(initial, **kwargs)
        self.initial = float(initial)

    def step(self, action: float) -> None:
        import torch

        self.initial += action
        self.env.logger.add_scalar("test_a", torch.tensor(233.0))
        self.env.logger.add_scalar("test_b", np.array(200))

    def get_state(self) -> float:
        return self.initial

    def done(self) -> bool:
        return self.initial % 1 > 0.5


class DummyStateInterpreter(StateInterpreter[float, float]):
    def interpret(self, state: float) -> float:
        return state

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(0, np.inf, shape=(), dtype=np.float32)


class DummyActionInterpreter(ActionInterpreter[float, int, float]):
    def interpret(self, state: float, action: int) -> float:
        return action / 100

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Discrete(5)


class RandomFivePolicy(BasePolicy):
    def forward(self, batch, state=None):
        return Batch(act=np.random.randint(5, size=len(batch)))

    def learn(self, batch):
        pass


def test_logger_with_env_wrapper():
    with DataQueue(list(range(20)), shuffle=False) as data_iterator:

        def env_wrapper_factory():
            return EnvWrapper(
                SimpleSimulator,
                DummyStateInterpreter(),
                DummyActionInterpreter(),
                data_iterator,
                logger=LogCollector(LogLevel.DEBUG),
            )

        # loglevel can be debugged here because metrics can all dump into csv
        # otherwise, csv writer might crash
        csv_writer = CsvWriter(Path(__file__).parent / ".output", loglevel=LogLevel.DEBUG)
        venv = vectorize_env(env_wrapper_factory, "shmem", 4, csv_writer)
        with venv.collector_guard():
            collector = Collector(RandomFivePolicy(), venv)
            collector.collect(n_episode=INF * len(venv))

    output_df = pd.read_csv(Path(__file__).parent / ".output" / "result.csv")
    assert len(output_df) == 20
    # obs has an increasing trend
    assert output_df["obs"].to_numpy()[:10].sum() < output_df["obs"].to_numpy()[10:].sum()
    assert (output_df["test_a"] == 233).all()
    assert (output_df["test_b"] == 200).all()
    assert "steps_per_episode" in output_df and "reward" in output_df
