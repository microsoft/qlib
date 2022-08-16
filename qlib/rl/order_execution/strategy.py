# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import collections
from abc import ABCMeta
from typing import Any, Dict, Generator, Tuple, cast

import pandas as pd

from qlib.backtest import CommonInfrastructure, Order
from qlib.backtest.decision import BaseTradeDecision, TradeDecisionWO, TradeRange, TradeRangeByTime
from qlib.backtest.utils import LevelInfrastructure, SAOE_DATA_KEY
from qlib.rl.data.exchange_wrapper import QlibIntradayBacktestData
from qlib.rl.order_execution.state import QlibBacktestAdapter, SAOEState
from qlib.rl.order_execution.utils import get_ticks_slice
from qlib.strategy.base import RLStrategy


class SAOEStrategy(RLStrategy, metaclass=ABCMeta):
    """RL-based strategies that use SAOEState as state."""

    def __init__(
        self,
        policy: object,  # TODO: add accurate typehint later.
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs: Any,
    ) -> None:
        super(SAOEStrategy, self).__init__(
            policy=policy,
            outer_trade_decision=outer_trade_decision,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )

        self.adapter_dict: Dict[tuple, QlibBacktestAdapter] = {}
        self._last_step_range = (0, 0)

    def _create_qlib_backtest_adapter(self, order: Order, trade_range: TradeRange) -> QlibBacktestAdapter:
        if not self.common_infra.has(SAOE_DATA_KEY):
            self.common_infra.reset_infra(**{SAOE_DATA_KEY: {}})

        # saoe_data can be considered as some type of cache. Use it to avoid unnecessary data reload.
        # The data for one order would be loaded only once. All strategies will reuse this data.
        saoe_data = self.common_infra.get(SAOE_DATA_KEY)
        if order.key not in saoe_data:
            data = self.trade_exchange.get_deal_price(
                stock_id=order.stock_id,
                start_time=order.start_time.replace(hour=0, minute=0, second=0),
                end_time=order.start_time.replace(hour=23, minute=59, second=59),
                direction=order.direction,
                method=None,
            )

            ticks_index = pd.DatetimeIndex(data.index)
            if isinstance(trade_range, TradeRangeByTime):
                ticks_for_order = get_ticks_slice(
                    ticks_index,
                    trade_range.start_time,
                    trade_range.end_time,
                    include_end=True,
                )
            else:
                ticks_for_order = None  # FIXME: implement this logic

            backtest_data = QlibIntradayBacktestData(
                order=order,
                exchange=self.trade_exchange,
                start_time=ticks_for_order[0],
                end_time=ticks_for_order[-1],
            )

            saoe_data[order.key] = (ticks_index, ticks_for_order, backtest_data)

        ticks_index, ticks_for_order, backtest_data = saoe_data[order.key]

        return QlibBacktestAdapter(
            order=order,
            executor=self.executor,
            exchange=self.trade_exchange,
            ticks_per_step=self.ticks_per_step,
            ticks_index=ticks_index,
            ticks_for_order=ticks_for_order,
            backtest_data=backtest_data,
        )

    def _update_last_step_range(self, step_range: Tuple[int, int]) -> None:
        self._last_step_range = step_range

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs: Any) -> None:
        super(SAOEStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)

        self.adapter_dict = {}
        self._last_step_range = (0, 0)

        if outer_trade_decision is not None and not outer_trade_decision.empty():
            trade_range = outer_trade_decision.trade_range
            assert trade_range is not None

            self.adapter_dict = {}
            for decision in outer_trade_decision.get_decision():
                order = cast(Order, decision)
                self.adapter_dict[order.key] = self._create_qlib_backtest_adapter(order, trade_range)

    def get_saoe_state_by_order(self, order: Order) -> SAOEState:
        return self.adapter_dict[order.key].saoe_state

    def post_upper_level_exe_step(self) -> None:
        for maintainer in self.adapter_dict.values():
            maintainer.generate_metrics_after_done()

    def post_exe_step(self, execute_result: list) -> None:
        last_step_length = self._last_step_range[1] - self._last_step_range[0]
        if last_step_length <= 0:
            assert not execute_result
            return

        results = collections.defaultdict(list)
        if execute_result is not None:
            for e in execute_result:
                results[e[0].key].append(e)

        for key, maintainer in self.adapter_dict.items():
            maintainer.update(results[key], self._last_step_range)


class DecomposedStrategy(SAOEStrategy):
    """Decomposed strategy that needs actions from outside to generate trade decisions."""

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, outer_trade_decision, level_infra, common_infra, **kwargs)

    def generate_trade_decision(self, execute_result: list = None) -> Generator[Any, Any, BaseTradeDecision]:
        # Once the following line is executed, this DecomposedStrategy (self) will be yielded to the outside
        # of the entire executor, and the execution will be suspended. When the execution is resumed by `send()`,
        # the sent item will be captured by `exec_vol`. The outside policy could communicate with the inner
        # level strategy through this way.
        exec_vol = yield self

        oh = self.trade_exchange.get_order_helper()
        order = oh.create(self._order.stock_id, exec_vol, self._order.direction)
        self._update_last_step_range(self.get_data_cal_avail_range(rtype="step"))

        return TradeDecisionWO([order], self)

    def alter_outer_trade_decision(self, outer_trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        return outer_trade_decision

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

        assert isinstance(outer_trade_decision, TradeDecisionWO)
        if outer_trade_decision is not None:
            order_list = outer_trade_decision.order_list
            assert len(order_list) == 1
            self._order = order_list[0]
