# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Generator, Optional

import pandas as pd
import qlib
from qlib.backtest import get_strategy_executor
from qlib.backtest.decision import Order
from qlib.backtest.executor import BaseExecutor, NestedExecutor, SimulatorExecutor
from qlib.config import REG_CN
from qlib.contrib.ops.high_freq import BFillNan, Cut, Date, DayCumsum, DayLast, FFillNan, IsInf, IsNull, Select
from qlib.rl.data.exchange_wrapper import QlibIntradayBacktestData
from qlib.rl.order_execution.state import SAOEState
from qlib.rl.order_execution.utils import get_ticks_slice
from qlib.rl.simulator import Simulator
from qlib.rl.strategy.saoe import SAOEStrategy


def init_qlib(qlib_config: dict) -> None:
    provider_uri_map = {
        "day": qlib_config["provider_uri_day"].as_posix(),
        "1min": qlib_config["provider_uri_1min"].as_posix(),
    }
    qlib.init(
        region=REG_CN,
        auto_mount=False,
        custom_ops=[DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut, DayCumsum],
        expression_cache=None,
        calendar_provider={
            "class": "LocalCalendarProvider",
            "module_path": "qlib.data.data",
            "kwargs": {
                "backend": {
                    "class": "FileCalendarStorage",
                    "module_path": "qlib.data.storage.file_storage",
                    "kwargs": {"provider_uri_map": provider_uri_map},
                },
            },
        },
        feature_provider={
            "class": "LocalFeatureProvider",
            "module_path": "qlib.data.data",
            "kwargs": {
                "backend": {
                    "class": "FileFeatureStorage",
                    "module_path": "qlib.data.storage.file_storage",
                    "kwargs": {"provider_uri_map": provider_uri_map},
                },
            },
        },
        provider_uri=provider_uri_map,
        kernels=1,
        redis_port=-1,
        clear_mem_cache=False,  # init_qlib will be called for multiple times. Keep the cache for improving performance
    )


def create_state_maintainer_recursive(
    executor: BaseExecutor,
    order: Order,
    backtest_data: QlibIntradayBacktestData,
    ticks_index: pd.DatetimeIndex,
    twap_price: float,
    ticks_for_order: pd.DatetimeIndex,
) -> None:
    if isinstance(executor, SimulatorExecutor):
        return
    else:
        assert isinstance(executor, NestedExecutor)

        if isinstance(executor.inner_strategy, SAOEStrategy):
            executor.inner_strategy.create_saoe_maintainer(
                order=order,
                executor=executor.inner_executor,
                backtest_data=backtest_data,
                time_per_step=executor.inner_executor.time_per_step,
                ticks_index=ticks_index,
                twap_price=twap_price,
                ticks_for_order=ticks_for_order,
            )

        create_state_maintainer_recursive(
            executor.inner_executor,
            order,
            backtest_data,
            ticks_index,
            twap_price,
            ticks_for_order,
        )


class SingleAssetOrderExecutionQlib(Simulator[Order, SAOEState, float]):
    """Single-asset order execution (SAOE) simulator which is implemented based on Qlib backtest tools.

    Parameters
    ----------
    order (Order):
        The seed to start an SAOE simulator is an order.
    qlib_config (dict):
        Configuration used to initialize Qlib.
    strategy_config (dict):
        Strategy configuration
    executor_config (dict):
        Executor configuration
    exchange_config (dict):
        Exchange configuration
    """

    def __init__(
        self,
        order: Order,
        qlib_config: dict,
        strategy_config: dict,
        executor_config: dict,
        exchange_config: dict,
    ) -> None:
        super().__init__(initial=order)

        assert order.start_time.date() == order.end_time.date(), "Start date and end date must be the same."

        init_qlib(qlib_config)

        self._collect_data_loop: Optional[Generator] = None
        self.reset(order, strategy_config, executor_config, exchange_config)

    def reset(
        self,
        order: Order,
        strategy_config: dict,
        executor_config: dict,
        exchange_config: dict,
    ) -> None:
        strategy, self._executor = get_strategy_executor(
            start_time=pd.Timestamp(order.start_time.date()),
            end_time=pd.Timestamp(order.start_time.date()) + pd.DateOffset(1),
            strategy=strategy_config,
            executor=executor_config,
            benchmark=order.stock_id,
            account=1e12,
            exchange_kwargs=exchange_config,
            pos_type="InfPosition",
        )
        assert isinstance(self._executor, NestedExecutor)
        strategy.reset(level_infra=self._executor.get_level_infra())

        exchange = self._executor.trade_exchange
        ticks_index = pd.DatetimeIndex([e[1] for e in list(exchange.quote_df.index)])
        ticks_for_order = get_ticks_slice(
            ticks_index,
            order.start_time,
            order.end_time,
            include_end=True,
        )
        backtest_data = QlibIntradayBacktestData(
            order=order,
            exchange=exchange,
            start_time=ticks_for_order[0],
            end_time=ticks_for_order[-1],
        )

        self.twap_price = backtest_data.get_deal_price().mean()

        self._collect_data_loop = self._executor.collect_data(strategy.generate_trade_decision(), level=0)
        assert isinstance(self._collect_data_loop, Generator)

        self._last_yielded_saoe_strategy = self._iter_strategy(action=None)

        create_state_maintainer_recursive(
            executor=self._executor,
            order=order,
            backtest_data=backtest_data,
            ticks_index=ticks_index,
            twap_price=self.twap_price,
            ticks_for_order=ticks_for_order,
        )

    def _iter_strategy(self, action: float = None) -> SAOEStrategy:
        """Iterate the _collect_data_loop until we get the next yield SAOEStrategy."""
        assert self._collect_data_loop is not None

        strategy = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        while not isinstance(strategy, SAOEStrategy):
            strategy = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        assert isinstance(strategy, SAOEStrategy)
        return strategy

    def step(self, action: float) -> None:
        """Execute one step or SAOE.

        Parameters
        ----------
        action (float):
            The amount you wish to deal. The simulator doesn't guarantee all the amount to be successfully dealt.
        """

        assert not self.done(), "Simulator has already done!"

        try:
            self._last_yielded_saoe_strategy = self._iter_strategy(action=action)
        except StopIteration:
            pass

        assert self._executor is not None

    def get_state(self) -> SAOEState:
        return self._last_yielded_saoe_strategy.maintainer.saoe_state

    def done(self) -> bool:
        return not self._executor.is_collecting
