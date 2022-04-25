# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import suppress
from random import randint, choice
from pathlib import Path

import re
import gym
import numpy as np
import pandas as pd
from tianshou.data import Collector, Batch
from tianshou.policy import BasePolicy

from qlib.log import set_log_with_config
from qlib.config import C
from qlib.rl.utils.env_wrapper import InfoDict
from qlib.rl.utils.log import LogCollector, CsvWriter, ConsoleWriter
from qlib.rl.utils.finite_env import finite_env_factory


class SimpleEnv(gym.Env[int, int]):
    def __init__(self):
        self.logger = LogCollector()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.step_count = 0
        return 0

    def step(self, action: int):
        self.logger.reset()

        self.logger.add_scalar("a", randint(1, 10))
        self.logger.add_array("b", pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

        if self.step_count >= 3:
            done = choice([False, True])
        else:
            done = False

        if 2 <= self.step_count <= 3:
            self.logger.add_scalar("c", randint(11, 20))

        self.step_count += 1

        return 1, 42., done, InfoDict(log=self.logger.logs(), aux_info={})


class AnyPolicy(BasePolicy):
    def forward(self, batch, state=None):
        return Batch(act=np.stack([1] * len(batch)))

    def learn(self, batch):
        pass


def test_simple_env_logger(caplog):
    set_log_with_config(C.logging_config)
    for venv_cls_name in ["dummy", "shmem", "subproc"]:
        writer = ConsoleWriter()
        csv_writer = CsvWriter(Path(__file__).parent / ".output")
        venv = finite_env_factory(lambda: SimpleEnv(), venv_cls_name, 4, [writer, csv_writer])
        collector = Collector(AnyPolicy(), venv)
        with suppress(StopIteration):
            collector.collect(n_episode=30)

    for line in caplog.text.splitlines():
        line = line.strip()
        if line:
            assert re.match(r".*reward 42\.0000 \(42.0000\)  a .* \(5\.\d+\)  c .* \((14|15|16)\.\d+\)", line)
