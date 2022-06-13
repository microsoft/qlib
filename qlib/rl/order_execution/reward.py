# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import cast

import numpy as np
from qlib.rl.reward import Reward

from .simulator_simple import SAOEState, SAOEMetrics

__all__ = ["PAPenaltyReward"]


class PAPenaltyReward(Reward[SAOEState]):
    """Encourage higher PAs, but penalize stacking all the amounts within a very short time.
    Formally, for each time step, the reward is :math:`(PA_t * vol_t / target - vol_t^2 * penalty)`.

    Parameters
    ----------
    penalty
        The penalty for large volume in a short time.
    """

    def __init__(self, penalty: float = 100.0):
        self.penalty = penalty

    def reward(self, simulator_state: SAOEState) -> float:
        whole_order = simulator_state.order.amount
        assert whole_order > 0
        latest_exec_record = cast(SAOEMetrics, simulator_state.history_steps.reset_index().iloc[-1].to_dict())
        pa = latest_exec_record["pa"] * latest_exec_record["amount"] / whole_order

        latest_interval = simulator_state.history_exec.loc[latest_exec_record["datetime"]:]
        penalty = -self.penalty * ((latest_interval["amount"] / whole_order) ** 2).sum()

        reward = pa + penalty

        # Throw error in case of NaN
        assert not (np.isnan(reward) or np.isinf(reward)), f"Invalid reward for simulator state: {simulator_state}"

        self.log("reward/pa", pa)
        self.log("reward/penalty", penalty)
        return reward
