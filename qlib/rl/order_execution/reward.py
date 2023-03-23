# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import cast

import numpy as np

from qlib.backtest.decision import OrderDir
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


class PPOReward(Reward[SAOEState]):
    """Reward proposed by paper "An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization".

    Parameters
    ----------
    max_step
        Maximum number of steps.
    start_time_index
        First time index that allowed to trade.
    end_time_index
        Last time index that allowed to trade.
    """

    def __init__(self, max_step: int, start_time_index: int = 0, end_time_index: int = 239) -> None:
        self.max_step = max_step
        self.start_time_index = start_time_index
        self.end_time_index = end_time_index

    def reward(self, simulator_state: SAOEState) -> float:
        if simulator_state.cur_step == self.max_step - 1 or simulator_state.position < 1e-6:
            if simulator_state.history_exec["deal_amount"].sum() == 0.0:
                vwap_price = cast(
                    float,
                    np.average(simulator_state.history_exec["market_price"]),
                )
            else:
                vwap_price = cast(
                    float,
                    np.average(
                        simulator_state.history_exec["market_price"],
                        weights=simulator_state.history_exec["deal_amount"],
                    ),
                )
            twap_price = simulator_state.backtest_data.get_deal_price().mean()

            if simulator_state.order.direction == OrderDir.SELL:
                ratio = vwap_price / twap_price if twap_price != 0 else 1.0
            else:
                ratio = twap_price / vwap_price if vwap_price != 0 else 1.0
            if ratio < 1.0:
                return -1.0
            elif ratio < 1.1:
                return 0.0
            else:
                return 1.0
        else:
            return 0.0
