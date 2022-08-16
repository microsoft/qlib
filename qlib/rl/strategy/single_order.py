# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.backtest import Order
from qlib.backtest.decision import BaseTradeDecision, OrderHelper, TradeDecisionWO, TradeRange
from qlib.strategy.base import BaseStrategy


class SingleOrderStrategy(BaseStrategy):
    """Strategy used to generate a trade decision with exactly one order.
    """
    def __init__(
        self,
        order: Order,
        trade_range: TradeRange,
        instrument: str,
    ) -> None:
        super().__init__()

        self._order = order
        self._trade_range = trade_range
        self._instrument = instrument

    def alter_outer_trade_decision(self, outer_trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        return outer_trade_decision

    def generate_trade_decision(self, execute_result: list = None) -> TradeDecisionWO:
        oh: OrderHelper = self.common_infra.get("trade_exchange").get_order_helper()
        order_list = [
            oh.create(
                code=self._instrument,
                amount=self._order.amount,
                direction=self._order.direction,
            ),
        ]
        return TradeDecisionWO(order_list, self, self._trade_range)
