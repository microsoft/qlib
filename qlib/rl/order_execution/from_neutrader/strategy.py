from abc import ABCMeta
from typing import Tuple

import pandas as pd

from qlib.backtest.decision import BaseTradeDecision, Order, OrderHelper, TradeDecisionWO, TradeRange
from qlib.backtest.utils import CommonInfrastructure
from qlib.rl.order_execution.from_neutrader.state import IntraDaySingleAssetDataSchema, SAOEEpisodicState
from qlib.rl.order_execution.from_neutrader.state_maintainer import StateMaintainer
from qlib.strategy.base import BaseStrategy


class RLStrategyBase(BaseStrategy, metaclass=ABCMeta):
    def post_exe_step(self, execute_result: list) -> None:
        """
        post process for each step of strategy this is design for RL Strategy,
        which require to update the policy state after each step

        NOTE: it is strongly coupled with RLNestedExecutor;
        """
        raise NotImplementedError("Please implement the `post_exe_step` method")


class DecomposedStrategy(RLStrategyBase):
    def __init__(self):
        super(DecomposedStrategy, self).__init__()

    def reset(self, outer_trade_decision: TradeDecisionWO = None, **kwargs) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)
        time_per_step = int(pd.Timedelta(self.trade_calendar.get_freq()) / pd.Timedelta("1min"))
        if outer_trade_decision is not None:
            self.maintainer = StateMaintainer(
                time_per_step,
                self.trade_calendar.get_all_time()[0],
                self.get_data_cal_avail_range(),
                self.trade_calendar.get_trade_step(),
                outer_trade_decision,
                self.trade_exchange,
            )

    def alter_outer_trade_decision(self, outer_trade_decision: BaseTradeDecision) -> BaseTradeDecision:
        return outer_trade_decision

    def post_exe_step(self, execute_result):
        self.maintainer.send_execute_result(execute_result)

    @property
    def sample_state_pair(self) -> Tuple[IntraDaySingleAssetDataSchema, SAOEEpisodicState]:
        assert len(self.maintainer.samples) == len(self.maintainer.states) == 1
        return (
            list(self.maintainer.samples.values())[0],
            list(self.maintainer.states.values())[0],
        )

    def generate_trade_decision(self, execute_result: list = None) -> BaseTradeDecision:
        # get a decision from the outmost loop
        exec_vol = yield self

        return TradeDecisionWO(
            self.maintainer.generate_orders(self.get_data_cal_avail_range(rtype="step"), [exec_vol]), self
        )


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
        trade_decision = TradeDecisionWO(order_list, self, self._trade_range)
        return trade_decision
