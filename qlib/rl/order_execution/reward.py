# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import cast

import numpy as np

from qlib.rl.order_execution.state import SAOEMetrics, SAOEState
from qlib.rl.reward import Reward

__all__ = ["PAPenaltyReward"]


class PAPenaltyReward(Reward[SAOEState]):
    """Encourage higher PAs, but penalize stacking all the amounts within a very short time.
    Formally, for each time step, the reward is :math:`(PA_t * vol_t / target - vol_t^2 * penalty)`.

    Parameters
    ----------
    penalty
        The penalty for large volume in a short time.
    scale
        The weight used to scale up or down the reward.
    """

    def __init__(self, penalty: float = 100.0, scale: float = 1.0) -> None:
        self.penalty = penalty
        self.scale = scale

    def reward(self, simulator_state: SAOEState) -> float:
        whole_order = simulator_state.order.amount
        assert whole_order > 0
        last_step = cast(SAOEMetrics, simulator_state.history_steps.reset_index().iloc[-1].to_dict())
        pa = last_step["pa"] * last_step["amount"] / whole_order

        # Inspect the "break-down" of the latest step: trading amount at every tick
        last_step_breakdown = simulator_state.history_exec.loc[last_step["datetime"] :]
        penalty = -self.penalty * ((last_step_breakdown["amount"] / whole_order) ** 2).sum()

        reward = pa + penalty

        # Throw error in case of NaN
        assert not (np.isnan(reward) or np.isinf(reward)), f"Invalid reward for simulator state: {simulator_state}"

        self.log("reward/pa", pa)
        self.log("reward/penalty", penalty)
        return reward * self.scale
