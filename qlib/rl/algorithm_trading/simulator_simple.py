# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, Optional

import numpy as np
import pandas as pd
from qlib.backtest.decision import Task
from qlib.constant import EPS, EPS_T
from qlib.rl.data.pickle_styled import DealPriceType, load_simple_intraday_backtest_data
from qlib.rl.simulator import Simulator
from qlib.rl.utils import LogLevel

from .state import SAATMetrics, SAATState

__all__ = ["SingleAssetAlgorithmTradingSimple"]


class SingleAssetAlgorithmTradingSimple(Simulator[Task, SAATState, float]):
    """Single-asset algorithm trading (SAAT) simulator.

    As there's no "calendar" in the simple simulator, ticks are used to trade.
    A tick is a record (a line) in the pickle-styled data file.
    Each tick is considered as a individual trading opportunity.
    If such fine granularity is not needed, use ``ticks_per_step`` to
    lengthen the ticks for each step.

    In each step, the traded amount are "equally" separated to each tick,
    then bounded by volume maximum execution volume (i.e., ``vol_threshold``),
    and if it's the last step, try to ensure all the amount to be executed.

    Parameters
    ----------
    task
        The seed to start an SAAT simulator is an task.
    data_granularity
        Number of ticks between consecutive data entries.
    ticks_per_step
        How many ticks per step.
    data_dir
        Path to load backtest data
    """

    history_exec: pd.DataFrame
    """All execution history at every possible time ticks. See :class:`SAATMetrics` for available columns.
    Index is ``datetime``.
    """

    history_steps: pd.DataFrame
    """Positions at each step. The position before first step is also recorded.
    See :class:`SAATMetrics` for available columns.
    Index is ``datetime``, which is the **starting** time of each step."""

    metrics: Optional[SAATMetrics]
    """Metrics. Only available when done."""

    ticks_index: pd.DatetimeIndex
    """All available ticks for the day (not restricted to task)."""

    ticks_for_trading: pd.DatetimeIndex
    """Ticks that is available for trading (sliced by task)."""

    def __init__(
        self,
        task: Task,
        data_dir: Path,
        fee_rate: float,
        data_granularity: int = 1,
        ticks_per_step: int = 30,
        deal_price_type: DealPriceType = "close",
    ) -> None:
        super().__init__(initial=task)

        assert ticks_per_step % data_granularity == 0

        self.task = task
        self.ticks_per_step: int = ticks_per_step // data_granularity
        self.deal_price_type = deal_price_type
        self.data_dir = data_dir
        self.fee_rate = fee_rate
        self.backtest_data = load_simple_intraday_backtest_data(
            self.data_dir,
            task.stock_id,
            pd.Timestamp(task.start_time.date()),
            self.deal_price_type,
            2,
        )

        self.ticks_index = self.backtest_data.get_time_index()

        # Get time index available for trading
        self.ticks_for_trading = self._get_ticks_slice(self.task.start_time, self.task.end_time)

        self.cur_time = self.ticks_for_trading[0]
        self.cur_step = 0
        # NOTE: astype(float) is necessary in some systems.
        # this will align the precision with `.to_numpy()` in `_split_exec_vol`
        self.current_cash = task.cash
        self.total_cash = task.cash
        self.position = 0

        metric_keys = list(SAATMetrics.__annotations__.keys())  # pylint: disable=no-member
        # NOTE: can empty dataframe contain index?
        self.history_exec = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.history_steps = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.metrics = None

        self.market_price: Optional[np.ndarray] = None
        self.market_vol: Optional[np.ndarray] = None
        self.market_vol_limit: Optional[np.ndarray] = None

    def step(self, action: str) -> None:
        """Execute one step or SAAT.

        Parameters
        ----------
        amount
            The amount you wish to deal. The simulator doesn't guarantee all the amount to be successfully dealt.
        """

        assert not self.done()
        self.market_price = self.market_vol = None  # avoid misuse
        trading_value = self.take_action(action)
        assert self.market_price is not None
        assert self.market_vol is not None

        if abs(self.position) < 1e-6:
            self.position = 0.0
        if abs(self.current_cash) < 1e-6:
            self.current_cash = 0.0
        if trading_value < 1e-6:
            trading_value = 0

        ret = self.position * (self.market_price[-1] - self.market_price[0])

        # Get time index available for this step
        time_index = self._get_ticks_slice(self.cur_time, self._next_time())

        self.history_exec = self._dataframe_append(
            self.history_exec,
            SAATMetrics(
                # It should have the same keys with SAOEMetrics,
                # but the values do not necessarily have the annotated type.
                # Some values could be vectorized (e.g., exec_vol).
                stock_id=self.task.stock_id,
                datetime=time_index,
                direction=2,  # other: 2
                market_volume=self.market_vol,
                market_price=self.market_price,
                action=action,
                cash=self.current_cash,
                total_cash=self.total_cash,
                position=self.position,
                trade_price=self.market_price[0],
                ret=ret,
                swap_value=trading_value,
            ),
        )

        self.history_steps = self._dataframe_append(
            self.history_steps,
            [
                SAATMetrics(
                    # It should have the same keys with SAOEMetrics,
                    # but the values do not necessarily have the annotated type.
                    # Some values could be vectorized (e.g., exec_vol).
                    stock_id=self.task.stock_id,
                    datetime=time_index,
                    direction=2,  # other: 2
                    market_volume=self.market_vol,
                    market_price=self.market_price,
                    action=action,
                    trading_value=trading_value,
                    cash=self.current_cash,
                    total_cash=self.total_cash,
                    position=self.position,
                    trade_price=self.market_price[0],
                    ret=ret,
                    swap_value=trading_value,
                )
            ],
        )

        if self.done():
            if self.env is not None:
                self.env.logger.add_any("history_steps", self.history_steps, loglevel=LogLevel.DEBUG)
                self.env.logger.add_any("history_exec", self.history_exec, loglevel=LogLevel.DEBUG)

            self.metrics = (
                SAATMetrics(
                    stock_id=self.task.stock_id,
                    datetime=time_index,
                    direction=self.task.direction,  # other: 2
                    market_volume=self.history_steps["market_vol"].sum(),
                    market_price=self.market_price[0],
                    action=action,
                    trading_value=self.history_steps["trading_value"].sum(),
                    cash=self.current_cash,
                    position=self.position,
                    trade_price=self.history_steps["trade_price"].mean(),
                    ret=self.history_steps["ret"].sum(),
                    swap_value=self.history_steps["trading_value"].sum(),
                ),
            )

            # NOTE (yuge): It looks to me that it's the "correct" decision to
            # put all the logs here, because only components like simulators themselves
            # have the knowledge about what could appear in the logs, and what's the format.
            # But I admit it's not necessarily the most convenient way.
            # I'll rethink about it when we have the second environment
            # Maybe some APIs like self.logger.enable_auto_log() ?

            if self.env is not None:
                for key, value in self.metrics.items():
                    if isinstance(value, float):
                        self.env.logger.add_scalar(key, value)
                    else:
                        self.env.logger.add_any(key, value)

        self.cur_time = self._next_time()
        self.cur_step += 1

    def get_state(self) -> SAATState:
        return SAATState(
            task=self.task,
            cur_time=self.cur_time,
            cur_step=self.cur_step,
            position=self.position,
            cash=self.current_cash,
            history_exec=self.history_exec,
            history_steps=self.history_steps,
            metrics=self.metrics,
            backtest_data=self.backtest_data,
            ticks_per_step=self.ticks_per_step,
            ticks_index=self.ticks_index,
            ticks_for_trading=self.ticks_for_trading,
        )

    def done(self) -> bool:
        return self.cur_time >= self.task.end_time

    def _next_time(self) -> pd.Timestamp:
        """The "current time" (``cur_time``) for next step."""
        # Look for next time on time index
        current_loc = self.ticks_index.get_loc(self.cur_time)
        next_loc = current_loc + self.ticks_per_step

        # Calibrate the next location to multiple of ticks_per_step.
        # This is to make sure that:
        # as long as ticks_per_step is a multiple of something, each step won't cross morning and afternoon.
        next_loc = next_loc - next_loc % self.ticks_per_step

        if next_loc < len(self.ticks_index) and self.ticks_index[next_loc] < self.task.end_time:
            return self.ticks_index[next_loc]
        else:
            return self.task.end_time

    def _cur_duration(self) -> pd.Timedelta:
        """The "duration" of this step (step that is about to happen)."""
        return self._next_time() - self.cur_time

    def take_action(self, action: str) -> np.ndarray:
        """
        Split the volume in each step into minutes, considering possible constraints.
        This follows TWAP strategy.
        """
        next_time = self._next_time()

        # get the backtest data for next interval
        self.market_vol = self.backtest_data.get_volume().loc[self.cur_time : next_time - EPS_T].to_numpy()
        self.market_price = self.backtest_data.get_deal_price().loc[self.cur_time : next_time - EPS_T].to_numpy()

        assert self.market_vol is not None and self.market_price is not None

        if next_time >= self.task.end_time and not self.position:
            trading_value = abs(self.market_price[-1] * self.position)
            self.current_cash += trading_value - self.fee_rate * trading_value
            self.position = 0

        if self.position == 0:
            if action == "long":
                trading_value = self.current_cash
                self.position = self.current_cash * (1 - self.fee_rate) / self.market_price[0]
                self.current_cash = 0
            elif action == "short":
                trading_value = self.current_cash
                self.position = -self.current_cash * (1 - self.fee_rate) / self.market_price[0]
                self.current_cash = 0
            else:
                trading_value = 0
        elif self.position > 0:
            if action == "long" or action == "hold":
                trading_value = 0
            else:
                trading_value = 2 * abs(self.market_price[0] * self.position)
                self.position = -self.position * (1 - self.fee_rate) ** 2
                self.current_cash = 0
        else:
            if action == "short" or action == "hold":
                trading_value = 0
            else:
                trading_value = 2 * abs(self.market_price[0] * self.position)
                self.position = -self.position * (1 - self.fee_rate) ** 2
                self.current_cash = 0

        return trading_value

    def _get_ticks_slice(self, start: pd.Timestamp, end: pd.Timestamp, include_end: bool = False) -> pd.DatetimeIndex:
        if not include_end:
            end = end - EPS_T
        return self.ticks_index[self.ticks_index.slice_indexer(start, end)]

    @staticmethod
    def _dataframe_append(df: pd.DataFrame, other: Any) -> pd.DataFrame:
        # dataframe.append is deprecated
        other_df = pd.DataFrame(other).set_index("datetime")
        other_df.index.name = "datetime"
        return pd.concat([df, other_df], axis=0)
