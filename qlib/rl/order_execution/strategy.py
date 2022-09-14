# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import collections
from types import GeneratorType
from typing import Any, Optional, Union, cast, Dict, Generator

import pandas as pd

from qlib.backtest import CommonInfrastructure, Order
from qlib.backtest.decision import BaseTradeDecision, TradeDecisionWO, TradeRange
from qlib.backtest.utils import LevelInfrastructure
from qlib.constant import ONE_MIN
from qlib.rl.data.exchange_wrapper import load_qlib_backtest_data
from qlib.rl.order_execution.state import SAOEStateAdapter, SAOEState
from qlib.strategy.base import RLStrategy


class SAOEStrategy(RLStrategy):
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

        self.adapter_dict: Dict[tuple, SAOEStateAdapter] = {}
        self._last_step_range = (0, 0)

    def _create_qlib_backtest_adapter(self, order: Order, trade_range: TradeRange) -> SAOEStateAdapter:
        backtest_data = load_qlib_backtest_data(order, self.trade_exchange, trade_range)

        return SAOEStateAdapter(
            order=order,
            executor=self.executor,
            exchange=self.trade_exchange,
            ticks_per_step=int(pd.Timedelta(self.trade_calendar.get_freq()) / ONE_MIN),
            backtest_data=backtest_data,
        )

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
                self.adapter_dict[order.key_by_day] = self._create_qlib_backtest_adapter(order, trade_range)

    def get_saoe_state_by_order(self, order: Order) -> SAOEState:
        return self.adapter_dict[order.key_by_day].saoe_state

    def post_upper_level_exe_step(self) -> None:
        for adapter in self.adapter_dict.values():
            adapter.generate_metrics_after_done()

    def post_exe_step(self, execute_result: Optional[list]) -> None:
        last_step_length = self._last_step_range[1] - self._last_step_range[0]
        if last_step_length <= 0:
            assert not execute_result
            return

        results = collections.defaultdict(list)
        if execute_result is not None:
            for e in execute_result:
                results[e[0].key_by_day].append(e)

        for key, adapter in self.adapter_dict.items():
            adapter.update(results[key], self._last_step_range)

    def generate_trade_decision(
        self,
        execute_result: list = None,
    ) -> Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]]:
        """
        For SAOEStrategy, we need to update the `self._last_step_range` every time a decision is generated.
        This operation should be invisible to developers, so we implement it in `generate_trade_decision()`
        The concrete logic to generate decisions should be implemented in `_generate_trade_decision()`.
        In other words, all subclass of `SAOEStrategy` should overwrite `_generate_trade_decision()` instead of
        `generate_trade_decision()`.
        """
        self._last_step_range = self.get_data_cal_avail_range(rtype="step")

        decision = self._generate_trade_decision(execute_result)
        if isinstance(decision, GeneratorType):
            decision = yield from decision

        return decision

    def _generate_trade_decision(self, execute_result: list = None) -> Generator[Any, Any, BaseTradeDecision]:
        raise NotImplementedError


class ProxySAOEStrategy(SAOEStrategy):
    """Proxy strategy that uses SAOEState. It is called a 'proxy' strategy because it does not make any decisions
    by itself. Instead, when the strategy is required to generate a decision, it will yield the environment's
    information and let the outside agents to make the decision. Please refer to `_generate_trade_decision` for
    more details.
    """

    def __init__(
        self,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, outer_trade_decision, level_infra, common_infra, **kwargs)

    def _generate_trade_decision(self, execute_result: list = None) -> Generator[Any, Any, BaseTradeDecision]:
        # Once the following line is executed, this ProxySAOEStrategy (self) will be yielded to the outside
        # of the entire executor, and the execution will be suspended. When the execution is resumed by `send()`,
        # the item will be captured by `exec_vol`. The outside policy could communicate with the inner
        # level strategy through this way.
        exec_vol = yield self

        oh = self.trade_exchange.get_order_helper()
        order = oh.create(self._order.stock_id, exec_vol, self._order.direction)

        return TradeDecisionWO([order], self)

    def reset(self, outer_trade_decision: BaseTradeDecision = None, **kwargs: Any) -> None:
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

        assert isinstance(outer_trade_decision, TradeDecisionWO)
        if outer_trade_decision is not None:
            order_list = outer_trade_decision.order_list
            assert len(order_list) == 1
            self._order = order_list[0]
