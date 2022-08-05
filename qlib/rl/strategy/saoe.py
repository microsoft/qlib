# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import collections
from abc import ABCMeta
from typing import Any, cast, Dict, Tuple

import pandas as pd
from qlib.backtest.decision import BaseTradeDecision, Order
from qlib.backtest.utils import CommonInfrastructure, LevelInfrastructure, SAOE_DATA_KEY
from qlib.rl.order_execution.state import SAOEState, SAOEStateMaintainer
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

        self.maintainer_dict: Dict[Tuple[str, int], SAOEStateMaintainer] = {}

    def _create_saoe_maintainer(self, order: Order) -> SAOEStateMaintainer:
        saoe_data = self.common_infra.get(SAOE_DATA_KEY)
        ticks_index, ticks_for_order, backtest_data = saoe_data[(order.stock_id, order.direction)]

        return SAOEStateMaintainer(
            order=order,
            executor=self.executor,
            exchange=self.trade_exchange,
            ticks_per_step=int(pd.Timedelta(self.trade_calendar.get_freq()) / pd.Timedelta("1min")),
            ticks_index=ticks_index,
            ticks_for_order=ticks_for_order,
            backtest_data=backtest_data,
        )

    def reset(
        self,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        outer_trade_decision: BaseTradeDecision = None,
        **kwargs,
    ) -> None:
        super(SAOEStrategy, self).reset(level_infra, common_infra, outer_trade_decision, **kwargs)

        self.maintainer_dict = {}
        for decision in outer_trade_decision.get_decision():
            order = cast(Order, decision)
            self.maintainer_dict[(order.stock_id, order.direction)] = self._create_saoe_maintainer(order)

    def get_saoe_state_by_order(self, order: Order) -> SAOEState:
        return self.maintainer_dict[(order.stock_id, order.direction)].saoe_state

    def post_upper_level_exe_step(self) -> None:
        for maintainer in self.maintainer_dict.values():
            maintainer.generate_metrics_after_done()

    def post_exe_step(self, execute_result: list) -> None:
        results = collections.defaultdict(list)
        if execute_result is not None:
            for e in execute_result:
                results[(e[0].stock_id, e[0].direction)].append(e)

        for (stock_id, direction), maintainer in self.maintainer_dict.items():
            maintainer.update(results[(stock_id, direction)])
