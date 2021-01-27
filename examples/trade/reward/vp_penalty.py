import numpy as np
from .base import Instant_Reward


class VP_Penalty_small(Instant_Reward):
    """Reward: (Abs(vv_ratio_t - 1) * 10000 - v_t^2 * penalty) / 100"""

    def __init__(self, config):
        self.penalty = config["penalty"]

    def get_reward(self, performance_raise, v_t, target, *args):
        """

        :param performance_raise: Abs(vv_ratio_t - 1) * 10000.
        :param target: Target volume
        :param v_t: The traded volume
        """
        assert target > 0
        reward = performance_raise * v_t / target
        reward -= self.penalty * (v_t / target) ** 2
        assert not (np.isnan(reward) or np.isinf(reward)), f"{performance_raise}, {v_t}, {target}"
        return reward / 100


class VP_Penalty_small_vec(VP_Penalty_small):
    def get_reward(self, performance_raise, v_t, target, *args):
        """

        :param performance_raise: Abs(vv_ratio_t - 1) * 10000.
        :param target: Target volume
        :param v_t: The traded volume
        """
        assert target > 0
        reward = performance_raise * v_t.sum() / target
        reward -= self.penalty * ((v_t / target) ** 2).sum()
        assert not (np.isnan(reward) or np.isinf(reward)), f"{performance_raise}, {v_t}, {target}"
        return reward / 100
