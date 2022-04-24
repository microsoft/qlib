# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from random import randint, choice

import gym
import pandas as pd

from qlib.rl.utils.env_wrapper import InfoDict
from qlib.rl.utils.log import LogCollector, LogWriter


class SimpleEnv(gym.Env[int, int]):
    def __init__(self):
        self.logger = LogCollector()

    def reset(self):
        self.step_count = 0
        return 0

    def step(self, action: int):
        self.logger.reset()

        self.logger.add_scalar('a', randint(1, 10))
        self.logger.add_array('b', pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))

        if self.step_count >= 3:
            done = choice([False, True])
        else:
            done = False

        if self.step_count >= 2:
            self.logger.add_scalar('c', randint(11, 20))

        self.step_count += 1

        return 1, 0., done, InfoDict(log=self.logger.logs(), aux_info={})
