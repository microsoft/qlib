import numpy as np
from .base import Instant_Reward


class PA_Penalty(Instant_Reward):
    """Reward: (Abs(tt_ratio_t - 1) * 10000 * v_t / target - v_t^2 * penalty) / 100"""

    def __init__(self, config):
        self.penalty = config["penalty"]

    def get_reward(self, performance_raise, v_t, target, PA_t, *args):
        reward = PA_t * v_t / target
        reward -= self.penalty * (v_t / target) ** 2
        return reward / 100
