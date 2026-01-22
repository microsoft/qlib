# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import cast

import numpy as np

from qlib.rl.algorithm_trading.state import SAATMetrics, SAATState
from qlib.rl.reward import Reward

__all__ = ["LongShortReward"]


class LongShortReward(Reward[SAATState]):
    """Encourage higher return considering transaction cost with both long and short operation.
    Formally, for each time step, the reward is :math:`(PA_t * vol_t / target - vol_t^2 * penalty)`.

    Parameters
    ----------
    trans_fee
        The cost for opening or closing a position.
    """

    def __init__(self, trans_fee: float = 0.0015, scale: float = 10.0) -> None:
        self.trans_fee = trans_fee
        self.scale = scale

    def reward(self, simulator_state: SAATState) -> float:
        last_step = cast(SAATMetrics, simulator_state.history_steps.reset_index().iloc[-1].to_dict())
        self.log("reward/ret_with_transfee", last_step["ret"])
        self.log("reward/trans_fee", last_step["swap_value"] * self.trans_fee)
        reward = last_step["ret"] / last_step["total_cash"]
        self.log("reward_total", reward)
        # Throw error in case of NaN
        assert not (np.isnan(reward) or np.isinf(reward)), f"Invalid reward for simulator state: {simulator_state}"

        return reward * self.scale
