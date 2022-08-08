# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Generator, Optional

import pandas as pd
import qlib
from qlib.backtest import get_strategy_executor
from qlib.backtest.decision import Order
from qlib.backtest.executor import NestedExecutor
from qlib.config import REG_CN
from qlib.contrib.ops.high_freq import BFillNan, Cut, Date, DayCumsum, DayLast, FFillNan, IsInf, IsNull, Select
from qlib.rl.order_execution.state import SAOEState
from qlib.rl.order_execution.strategy import SAOEStrategy
from qlib.rl.simulator import Simulator


def init_qlib(qlib_config: dict) -> None:
    """Initialize necessary resource to launch the workflow, including data direction, feature columns, etc..

    Parameters
    ----------
    qlib_config:
        Qlib configuration.

        Example:
            {
                "provider_uri_day": DATA_ROOT_DIR / "qlib_1d",
                "provider_uri_1min": DATA_ROOT_DIR / "qlib_1min",
                "feature_root_dir": DATA_ROOT_DIR / "qlib_handler_stock",
                "feature_columns_today": [
                    "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume",
                    "$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5",
                ],
                "feature_columns_yesterday": [
                    "$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1", "$bid_1", "$ask_1", "$volume_1",
                    "$bidV_1", "$bidV1_1", "$bidV3_1", "$bidV5_1", "$askV_1", "$askV1_1", "$askV3_1", "$askV5_1",
                ],
            }
    """

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


class SingleAssetOrderExecutionQlib(Simulator[Order, SAOEState, float]):
    """Single-asset order execution (SAOE) simulator which is implemented based on Qlib backtest tools.

    Parameters
    ----------
    order
        The seed to start an SAOE simulator is an order.
    strategy_config
        Strategy configuration
    executor_config
        Executor configuration
    exchange_config
        Exchange configuration
    qlib_config
        Configuration used to initialize Qlib. If it is None, Qlib will not be initialized.
    """

    def __init__(
        self,
        order: Order,
        strategy_config: dict,
        executor_config: dict,
        exchange_config: dict,
        qlib_config: dict = None,
    ) -> None:
        super().__init__(initial=order)

        assert order.start_time.date() == order.end_time.date(), "Start date and end date must be the same."

        self._collect_data_loop: Optional[Generator] = None
        self.reset(order, strategy_config, executor_config, exchange_config, qlib_config)

    def reset(
        self,
        order: Order,
        strategy_config: dict,
        executor_config: dict,
        exchange_config: dict,
        qlib_config: dict = None,
    ) -> None:
        if qlib_config is not None:
            init_qlib(qlib_config)

        strategy, self._executor = get_strategy_executor(
            start_time=order.start_time.replace(hour=0, minute=0, second=0),
            end_time=order.start_time.replace(hour=0, minute=0, second=0) + pd.DateOffset(1),
            strategy=strategy_config,
            executor=executor_config,
            benchmark=order.stock_id,
            account=1e12,
            exchange_kwargs=exchange_config,
            pos_type="InfPosition",
        )

        assert isinstance(self._executor, NestedExecutor)
        strategy.reset(level_infra=self._executor.get_level_infra())  # TODO: check if we could remove this

        self._collect_data_loop = self._executor.collect_data(strategy.generate_trade_decision(), level=0)
        assert isinstance(self._collect_data_loop, Generator)

        self._last_yielded_saoe_strategy = self._iter_strategy(action=None)

        self._order = order

    @property
    def twap_price(self) -> float:
        return self._last_yielded_saoe_strategy.adapter_dict[self._order.key].twap_price

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
        return self._last_yielded_saoe_strategy.get_saoe_state_by_order(self._order)

    def done(self) -> bool:
        return not self._executor.is_collecting
