from typing import List, Optional, Tuple

from qlib.backtest.decision import BaseTradeDecision, Order, OrderHelper, TradeDecisionWO, TradeRange
from qlib.backtest.utils import CommonInfrastructure
from qlib.strategy.base import BaseStrategy


class DecomposedStrategy(BaseStrategy):
    def __init__(self) -> None:
        super(DecomposedStrategy, self).__init__()

        self.execute_order: Optional[Order] = None
        self.execute_result: List[Tuple[Order, float, float, float]] = []

    def generate_trade_decision(self, execute_result: list = None) -> BaseTradeDecision:
        exec_vol = yield self

        oh = self.trade_exchange.get_order_helper()
        order = oh.create(self._order.stock_id, exec_vol, self._order.direction)

        self.execute_order = order

        return TradeDecisionWO([order], self)

    def alter_outer_trade_decision(self, outer_trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        return outer_trade_decision

    def receive_execute_result(self, execute_result: list) -> None:
        self.execute_result = execute_result

    def reset(self, outer_trade_decision: TradeDecisionWO = None, **kwargs) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            order_list = outer_trade_decision.order_list
            assert len(order_list) == 1
            self._order = order_list[0]


class SingleOrderStrategy(BaseStrategy):
    # this logic is copied from FileOrderStrategy
    def __init__(
        self,
        common_infra: CommonInfrastructure,
        order: Order,
        trade_range: TradeRange,
        instrument: str,
    ) -> None:
        super().__init__(common_infra=common_infra)
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
                direction=Order.parse_dir(self._order.direction),
            )
        ]
        return TradeDecisionWO(order_list, self, self._trade_range)
