import numpy as np
from .base import Abs_Reward


class PPO_Reward(Abs_Reward):
    """The reward function defined in IJCAI 2020"""

    def __init__(self, *args):
        pass

    def isinstant(self):
        return False

    def get_reward(self, performace_raise, ffr, this_tt_ratio, is_buy):
        if is_buy:
            this_tt_ratio = 1 / this_tt_ratio
        if this_tt_ratio < 1:
            return -1.0
        elif this_tt_ratio < 1.1:
            return 0.0
        else:
            return 1.0
