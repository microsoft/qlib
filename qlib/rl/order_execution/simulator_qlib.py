# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Generator, Optional

import pandas as pd
from qlib.backtest import get_strategy_executor
from qlib.backtest.decision import Order
from qlib.backtest.executor import NestedExecutor
from qlib.rl.integration.feature import init_qlib
from qlib.rl.order_execution.state import SAOEState
from qlib.rl.order_execution.strategy import SAOEStrategy
from qlib.rl.simulator import Simulator


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
            init_qlib(qlib_config, part="skip")

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
