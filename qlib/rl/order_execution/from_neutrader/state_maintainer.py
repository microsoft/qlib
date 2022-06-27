from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.exchange import Exchange
from qlib.constant import REG_CN
from qlib.rl.order_execution.from_neutrader.feature import fetch_features
from qlib.rl.order_execution.from_neutrader.state import FlowDirection, IntraDaySingleAssetDataSchema, SAOEEpisodicState
from qlib.utils.time import get_day_min_idx_range


class StateMaintainer:
    """
    Maintain neutrader states taking qlib trade decisions as input.

    Example usage::

        maintainer = StateMaintainer(...)  # in reset
        maintainer.send_execute_result(execute_result)  # in step
        # do something here
        maintainer.generate_orders(self.get_data_cal_avail_range(rtype='step'), exec_vols)

    The states can be accessed via ``maintianer.states`` and ``maintainer.samples``.
    """

    def __init__(
        self,
        time_per_step: int,
        date: pd.Timestamp,
        full_trade_range: Tuple[int, int],
        current_step: int,
        outer_trade_decision: TradeDecisionWO,
        trade_exchange: Exchange,
    ) -> None:
        # The parameters look very ad-hoc right now
        self.states: Dict[Tuple[str, OrderDir], SAOEEpisodicState] = {}  # explicitly make it ordered
        self.samples: Dict[Tuple[str, OrderDir], IntraDaySingleAssetDataSchema] = {}
        self.time_per_step: int = time_per_step
        self.start_time, self.end_time = full_trade_range
        self.end_time += 1  # plus 1 to align with the semantics in neutrader
        self.date: pd.Timestamp = date
        self.last_step_length: int = -1
        self.last_step_range: Optional[Tuple[int, int]] = None

        self.order_list: List[Order] = outer_trade_decision.order_list
        self.trade_exchange: Exchange = trade_exchange

        self.num_step = (
                            self.end_time - (self.start_time - self.start_time % self.time_per_step) - 1
                        ) // self.time_per_step + 1

        for order in self.order_list:
            sample = self._fetch_sample_data(order)
            state = self._create_single_ep_state(sample, current_step)
            self.samples[order.stock_id, order.direction] = sample
            self.states[order.stock_id, order.direction] = state

    def _fetch_sample_data(self, order: Order) -> IntraDaySingleAssetDataSchema:
        start_time = self.date.replace(hour=0, minute=0, second=0)
        end_time = self.date.replace(hour=23, minute=59, second=59)
        deal_price = self.trade_exchange.get_deal_price(
            stock_id=order.stock_id, start_time=start_time, end_time=end_time, direction=order.direction, method=None,
        )
        backtest_data = fetch_features(order.stock_id, self.date, backtest=True)
        # HACK: close means deal price here. The logic is implemented in qlib.
        backtest_data["$close"] = deal_price.to_series().to_numpy()
        feature_today = fetch_features(order.stock_id, self.date)
        feature_yesterday = fetch_features(order.stock_id, self.date, yesterday=True)
        return IntraDaySingleAssetDataSchema(
            date=self.date.date(),
            stock_id=order.stock_id,
            start_time=self.start_time,
            end_time=self.end_time,
            target=max(order.amount, 0.0),  # prevent target to go to -eps
            flow_dir=FlowDirection.LIQUIDATE if order.direction == 0 else FlowDirection.ACQUIRE,
            raw=backtest_data,
            processed=feature_today,
            processed_prev=feature_yesterday,
        )

    def _create_single_ep_state(self, sample: IntraDaySingleAssetDataSchema, cur_step: int) -> SAOEEpisodicState:
        market_price = sample.raw["$close"].values
        market_vol = sample.raw["$volume"].values
        target = sample.target

        # NOTE: Previously, market_price and market_vol are passed into the state initialization directly. Therefore,
        # the segment of market_price and market_vol are used instead of the lambda function here using the whole price
        # and vol data.
        # This refactoring is ONLY EQUIVALENT WHEN start_time/end_time passed into state is equal to
        # sample.start_time/end_time.
        # If one can confirm that these two are always the same, delete this note, please.
        state = SAOEEpisodicState(
            self.start_time,
            self.end_time,
            self.time_per_step,
            None,
            lambda x: market_price,
            lambda: market_vol,
            None,
            None,
            1,
            target,
            target,
            sample.flow_dir,
        )
        state.cur_step = cur_step
        assert state.cur_step == 0
        return state

    def _update_single_ep_state(
        self, state: SAOEEpisodicState, execute_result: List[Order], length: Optional[int] = None
    ) -> None:
        if length is not None:
            exec_vol = np.zeros(length)
            for order, _, __, ___ in execute_result:
                idx, _ = get_day_min_idx_range(order.start_time, order.end_time, "1min", REG_CN)
                exec_vol[idx - self.last_step_range[0]] = order.deal_amount
        else:
            exec_vol = np.array([order.deal_amount for order, _, __, ___ in execute_result])

        # sometimes exec_vol gets too large due to the rounding in exchange
        # scale the execution volume so that position won't go below 0
        # actually this case is very rare
        if exec_vol.sum() > state.position and exec_vol.sum() > 0:
            assert exec_vol.sum() < state.position + 1, f"{exec_vol} too large for {state}"
            exec_vol *= state.position / (exec_vol.sum())

        state.step(exec_vol)

    def create_sub_order(self, exec_vol: float, original_order: Order) -> Order:
        oh = self.trade_exchange.get_order_helper()
        return oh.create(original_order.stock_id, exec_vol, original_order.direction)

    def send_execute_result(self, execute_result: Optional[List[Any]]) -> None:
        if self.last_step_length < 0:
            assert not execute_result
            return
        orders = defaultdict(list)
        if execute_result is not None:
            for e in execute_result:
                orders[e[0].stock_id, e[0].direction].append(e)
        for (stock_id, direction), state in self.states.items():
            self._update_single_ep_state(state, orders[stock_id, direction], self.last_step_length)

    def generate_orders(self, step_trade_range: Tuple[int, int], exec_vols: List[float]) -> List[Order]:
        order_list = []

        assert len(exec_vols) == len(self.order_list)
        for v, o in zip(exec_vols, self.order_list):
            if v > 0:
                order_list.append(self.create_sub_order(v, o))

        step_start_time, step_end_time = step_trade_range  # inclusive
        step_end_time += 1

        self.last_step_length = step_end_time - step_start_time
        self.last_step_range = (step_start_time, step_end_time)

        return order_list
