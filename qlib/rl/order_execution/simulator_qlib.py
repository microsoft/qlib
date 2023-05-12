# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Generator, List, Optional

import pandas as pd

from qlib.backtest import collect_data_loop, get_strategy_executor
from qlib.backtest.decision import BaseTradeDecision, Order, TradeRangeByTime
from qlib.backtest.executor import NestedExecutor
from qlib.rl.data.integration import init_qlib
from qlib.rl.simulator import Simulator
from .state import SAOEState
from .strategy import SAOEStateAdapter, SAOEStrategy


class SingleAssetOrderExecution(Simulator[Order, SAOEState, float]):
    """Single-asset order execution (SAOE) simulator which is implemented based on Qlib backtest tools.

    Parameters
    ----------
    order
        The seed to start an SAOE simulator is an order.
    executor_config
        Executor configuration
    exchange_config
        Exchange configuration
    qlib_config
        Configuration used to initialize Qlib. If it is None, Qlib will not be initialized.
    cash_limit:
        Cash limit.
    """

    def __init__(
        self,
        order: Order,
        executor_config: dict,
        exchange_config: dict,
        qlib_config: dict | None = None,
        cash_limit: float | None = None,
    ) -> None:
        super().__init__(initial=order)

        assert order.start_time.date() == order.end_time.date(), "Start date and end date must be the same."

        strategy_config = {
            "class": "SingleOrderStrategy",
            "module_path": "qlib.rl.strategy.single_order",
            "kwargs": {
                "order": order,
                "trade_range": TradeRangeByTime(order.start_time.time(), order.end_time.time()),
            },
        }

        self._collect_data_loop: Optional[Generator] = None
        self.reset(order, strategy_config, executor_config, exchange_config, qlib_config, cash_limit)

    def reset(
        self,
        order: Order,
        strategy_config: dict,
        executor_config: dict,
        exchange_config: dict,
        qlib_config: dict | None = None,
        cash_limit: Optional[float] = None,
    ) -> None:
        if qlib_config is not None:
            init_qlib(qlib_config)

        strategy, self._executor = get_strategy_executor(
            start_time=order.date,
            end_time=order.date + pd.DateOffset(1),
            strategy=strategy_config,
            executor=executor_config,
            benchmark=order.stock_id,
            account=cash_limit if cash_limit is not None else int(1e12),
            exchange_kwargs=exchange_config,
            pos_type="Position" if cash_limit is not None else "InfPosition",
        )

        assert isinstance(self._executor, NestedExecutor)

        self.report_dict: dict = {}
        self.decisions: List[BaseTradeDecision] = []
        self._collect_data_loop = collect_data_loop(
            start_time=order.date,
            end_time=order.date,
            trade_strategy=strategy,
            trade_executor=self._executor,
            return_value=self.report_dict,
        )
        assert isinstance(self._collect_data_loop, Generator)

        self.step(action=None)

        self._order = order

    def _get_adapter(self) -> SAOEStateAdapter:
        return self._last_yielded_saoe_strategy.adapter_dict[self._order.key_by_day]

    @property
    def twap_price(self) -> float:
        return self._get_adapter().twap_price

    def _iter_strategy(self, action: Optional[float] = None) -> SAOEStrategy:
        """Iterate the _collect_data_loop until we get the next yield SAOEStrategy."""
        assert self._collect_data_loop is not None

        obj = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        while not isinstance(obj, SAOEStrategy):
            if isinstance(obj, BaseTradeDecision):
                self.decisions.append(obj)
            obj = next(self._collect_data_loop) if action is None else self._collect_data_loop.send(action)
        assert isinstance(obj, SAOEStrategy)
        return obj

    def step(self, action: Optional[float]) -> None:
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
        return self._get_adapter().saoe_state

    def done(self) -> bool:
        return self._executor.finished()
