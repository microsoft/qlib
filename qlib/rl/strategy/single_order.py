# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.backtest import Order
from qlib.backtest.decision import OrderHelper, TradeDecisionWO, TradeRange
from qlib.strategy.base import BaseStrategy


class SingleOrderStrategy(BaseStrategy):
    """Strategy used to generate a trade decision with exactly one order."""

    def __init__(
        self,
        order: Order,
        trade_range: TradeRange = None,
    ) -> None:
        super().__init__()

        self._order = order
        self._trade_range = trade_range

    def generate_trade_decision(self, execute_result: list = None) -> TradeDecisionWO:
        oh: OrderHelper = self.common_infra.get("trade_exchange").get_order_helper()
        order_list = [
            oh.create(
                code=self._order.stock_id,
                amount=self._order.amount,
                direction=self._order.direction,
            ),
        ]
        return TradeDecisionWO(order_list, self, self._trade_range)
