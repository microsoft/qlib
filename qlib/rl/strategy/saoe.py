# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABCMeta
from typing import Optional

import pandas as pd
from qlib.backtest.decision import BaseTradeDecision, Order
from qlib.backtest.executor import BaseExecutor
from qlib.backtest.utils import CommonInfrastructure, LevelInfrastructure
from qlib.rl.data.exchange_wrapper import QlibIntradayBacktestData
from qlib.rl.order_execution.state_maintainer import SAOEStateMaintainer
from qlib.strategy.base import RLStrategy


class SAOEStrategy(RLStrategy, metaclass=ABCMeta):
    """RL-based strategies that use SAOEState as state."""

    def __init__(
        self,
        policy,
        outer_trade_decision: BaseTradeDecision = None,
        level_infra: LevelInfrastructure = None,
        common_infra: CommonInfrastructure = None,
        **kwargs,
    ) -> None:
        super(SAOEStrategy, self).__init__(policy, outer_trade_decision, level_infra, common_infra, **kwargs)

        self.maintainer: Optional[SAOEStateMaintainer] = None

    def create_saoe_maintainer(
        self,
        order: Order,
        executor: BaseExecutor,
        backtest_data: QlibIntradayBacktestData,
        time_per_step: str,
        ticks_index: pd.DatetimeIndex,
        twap_price: float,
        ticks_for_order: pd.DatetimeIndex,
    ) -> None:
        self.maintainer = SAOEStateMaintainer(
            order=order,
            executor=executor,
            backtest_data=backtest_data,
            time_per_step=time_per_step,
            ticks_index=ticks_index,
            twap_price=twap_price,
            ticks_for_order=ticks_for_order,
        )

    def post_upper_level_exe_step(self) -> None:
        self.maintainer.generate_metrics_after_done()

    def post_exe_step(self, execute_result: list) -> None:
        self.maintainer.update(
            execute_result=execute_result,
        )
