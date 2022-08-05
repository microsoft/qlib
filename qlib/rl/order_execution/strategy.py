# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import collections
from abc import ABCMeta
from typing import Any, cast, Dict, Generator, Tuple

import pandas as pd
from qlib.backtest import CommonInfrastructure, Order
from qlib.backtest.decision import BaseTradeDecision, TradeDecisionWO
from qlib.backtest.utils import LevelInfrastructure, SAOE_DATA_KEY
from qlib.rl.order_execution.state import QlibBacktestAdapter, SAOEState
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
        super(SAOEStrategy, self).__init__(policy, outer_trade_decision, level_infra, common_infra, **kwargs)

        self.adapter_dict: Dict[Tuple[str, int], QlibBacktestAdapter] = {}

    def _create_qlib_backtest_adapter(self, order: Order) -> QlibBacktestAdapter:
        saoe_data = self.common_infra.get(SAOE_DATA_KEY)
        ticks_index, ticks_for_order, backtest_data = saoe_data[(order.stock_id, order.direction)]

        return QlibBacktestAdapter(
            order=order,
            executor=self.executor,
            exchange=self.trade_exchange,
            ticks_per_step=int(pd.Timedelta(self.trade_calendar.get_freq()) / pd.Timedelta("1min")),
            ticks_index=ticks_index,
            ticks_for_order=ticks_for_order,
            backtest_data=backtest_data,
        )

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs: Any) -> None:
        super(SAOEStrategy, self).reset(outer_trade_decision=outer_trade_decision, **kwargs)

        if outer_trade_decision is not None:
            self.adapter_dict = {}
            for decision in outer_trade_decision.get_decision():
                order = cast(Order, decision)
                self.adapter_dict[(order.stock_id, order.direction)] = self._create_qlib_backtest_adapter(order)

    def get_saoe_state_by_order(self, order: Order) -> SAOEState:
        return self.adapter_dict[(order.stock_id, order.direction)].saoe_state

    def post_upper_level_exe_step(self) -> None:
        for maintainer in self.adapter_dict.values():
            maintainer.generate_metrics_after_done()

    def post_exe_step(self, execute_result: list) -> None:
        results = collections.defaultdict(list)
        if execute_result is not None:
            for e in execute_result:
                results[(e[0].stock_id, e[0].direction)].append(e)

        for (stock_id, direction), maintainer in self.adapter_dict.items():
            maintainer.update(results[(stock_id, direction)])


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
