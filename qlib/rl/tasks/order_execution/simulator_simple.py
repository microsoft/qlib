# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from qlib.backtest import Order
from qlib.constant import EPS
from qlib.rl.simulator import Simulator
from qlib.rl.tasks.data.pickle_styled import (
    IntradayBacktestData, get_intraday_backtest_data, DealPriceType
)


class SAOEState(NamedTuple):
    """
    Base class for episodic states.
    """

    order: Order                        # the order we are dealing with
    cur_time: pd.Timestamp              # current time, e.g., 9:30
    elapsed_ticks: int                  # current time in data index, e.g., in [0, 239]
    traded_tricks: int                  # elapsed ticks in the period of time defined by order
    total_ticks: int                    # length of data time index sliced in order period, e.g., 240
    position: float                     # current remaining volume to execute
    position_history: list[float]       # position history, the initial position included
    exec_history: pd.Series             # see :attr:`SingleAssetOrderExecution.exec_history`

    # Backtest data is included in the state.
    # Actually, only the time index of this data is needed, at this moment.
    # I include the full data so that algorithms (e.g., VWAP) that relies on the raw data can be implemented.

    backtest_data: IntradayBacktestData # backtest data. interpreter should be careful not to leak feature

    # All possible trading ticks in all day (defined in data). e.g., [9:30, 9:31, ..., 14:59]
    ticks_index: pd.DatetimeIndex


class SingleAssetOrderExecution(Simulator[Order, SAOEState, float]):
    """Single-asset order execution (SAOE) simulator.

    Parameters
    ----------
    initial
        The seed to start an SAOE simulator is an order.
    time_per_step
        Elapsed time per step.
    data_dir
        Path to load backtest data
    vol_threshold
        Maximum execution volume (divided by market execution volume).
    """

    exec_history: pd.Series | None
    """All execution volumes at every possible time ticks."""
    position_history: list[float]
    """Positions at each step. The position before first step is also recorded."""

    def __init__(self, order: Order, data_dir: Path,
                 time_per_step: str = '30min',
                 deal_price_type: DealPriceType = 'close',
                 vol_threshold: float | None = None) -> None:
        self.order = order
        self.time_per_step = pd.Timedelta(time_per_step)
        self.deal_price_type = deal_price_type
        self.vol_threshold = vol_threshold
        self.cur_time = order.start_time
        self.data_dir = data_dir
        self.backtest_data = get_intraday_backtest_data(
            self.data_dir,
            order.stock_id,
            pd.Timestamp(order.start_time.date),
            self.deal_price_type,
            order.direction
        )

        self.position = order.amount

        self.exec_history = None
        self.position_history = [self.position]

        self.market_price: np.ndarray | None = None
        self.market_vol: np.ndarray | None = None
        self.market_vol_limit: np.ndarray | None = None

    def step(self, amount: float) -> None:
        """Execute one step or SAOE.

        Parameters
        ----------
        amount
            The amount you wish to deal. The simulator doesn't guarantee all the amount to be successfully dealt.
        """

        exec_vol = self._split_exec_vol(amount)

        self.position -= exec_vol.sum()
        if self.position < -EPS and not (exec_vol < -EPS).any():
            raise ValueError(f'Execution volume is invalid: {exec_vol} (position = {self.position})')
        self.position_history.append(self.position)
        self.cur_time = self._next_time()

        exec_vol_series = pd.Series(
            data=exec_vol,
            index=self.backtest_data.get_time_index().slice_indexer(self.cur_time, self._next_time())
        )

        if self.exec_history is None:
            self.exec_history = exec_vol_series
        else:
            self.exec_history = pd.concat([self.exec_history, exec_vol_series])

        raise NotImplementedError()

    def get_state(self) -> SAOEState:
        return SAOEState(
            order=self.order,
            cur_time=self.cur_time,
            elapsed_ticks=int(np.sum(self.cur_time > self.backtest_data.get_time_index())),
            traded_tricks=len(self.exec_history),
            total_ticks=len(self.backtest_data.loc[self.order.start_time:self.order.end_time]),
            position=self.position,
            position_history=self.position_history,
            backtest_data=self.backtest_data,
            ticks_index=self.backtest_data.get_time_index()
        )

    def done(self) -> bool:
        return self.position < EPS or self.cur_time >= self.order.end_time

    def _next_time(self) -> pd.Timestamp:
        """The "current time" (``cur_time``) for next step."""
        return min(self.order.end_time, self.cur_time + self.time_per_step)

    def _cur_duration(self) -> pd.Timedelta:
        """The "duration" of this step (step that is about to happen)."""
        return self._next_time() - self.cur_time

    def _split_exec_vol(self, exec_vol_sum: float) -> np.ndarray:
        """
        Split the volume in each step into minutes, considering possible constraints.
        This follows TWAP strategy.
        """
        next_time = self._next_time()
        ONE_SEC = pd.Timedelta('1s')  # use 1 second to exclude the right interval point

        # get the backtest data for next interval
        backtest_interval = self.backtest_data.loc[self.cur_time:next_time - ONE_SEC]
        self.market_vol = backtest_interval['$volume0'].to_numpy()
        self.market_price = self.backtest_data.get_deal_price(self.order.direction) \
            .loc[self.cur_time:next_time - ONE_SEC].to_numpy()

        # split the volume equally into each minute
        exec_vol = np.repeat(exec_vol_sum / len(backtest_interval), len(backtest_interval))

        # apply the volume threshold
        market_vol_limit = self.vol_threshold * self.market_vol if self.vol_threshold is not None else np.inf
        exec_vol = np.minimum(exec_vol, market_vol_limit)

        # Complete all the order amount at the last moment.
        if next_time == self.order.end_time:
            exec_vol[-1] += self.position - exec_vol.sum()
            exec_vol = np.minimum(exec_vol, market_vol_limit)

        return exec_vol
