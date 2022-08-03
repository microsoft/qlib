# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Generator

from qlib.backtest.decision import BaseTradeDecision, TradeDecisionWO
from qlib.backtest.utils import CommonInfrastructure, LevelInfrastructure
from qlib.rl.strategy.saoe import SAOEStrategy


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

    def reset(self, outer_trade_decision: TradeDecisionWO = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            order_list = outer_trade_decision.order_list
            assert len(order_list) == 1
            self._order = order_list[0]
