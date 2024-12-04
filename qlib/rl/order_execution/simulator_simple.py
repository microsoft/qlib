# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, cast, List, Optional

import numpy as np
import pandas as pd

from pathlib import Path
from qlib.backtest.decision import Order, OrderDir
from qlib.constant import EPS, EPS_T, float_or_ndarray
from qlib.rl.data.base import BaseIntradayBacktestData
from qlib.rl.data.native import DataframeIntradayBacktestData, load_handler_intraday_processed_data
from qlib.rl.data.pickle_styled import load_simple_intraday_backtest_data
from qlib.rl.simulator import Simulator
from qlib.rl.utils import LogLevel
from .state import SAOEMetrics, SAOEState

__all__ = ["SingleAssetOrderExecutionSimple"]


class SingleAssetOrderExecutionSimple(Simulator[Order, SAOEState, float]):
    """Single-asset order execution (SAOE) simulator.

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
    order
        The seed to start an SAOE simulator is an order.
    data_dir
        Path to load backtest data.
    feature_columns_today
        Columns of today's feature.
    feature_columns_yesterday
        Columns of yesterday's feature.
    data_granularity
        Number of ticks between consecutive data entries.
    ticks_per_step
        How many ticks per step.
    vol_threshold
        Maximum execution volume (divided by market execution volume).
    """

    history_exec: pd.DataFrame
    """All execution history at every possible time ticks. See :class:`SAOEMetrics` for available columns.
    Index is ``datetime``.
    """

    history_steps: pd.DataFrame
    """Positions at each step. The position before first step is also recorded.
    See :class:`SAOEMetrics` for available columns.
    Index is ``datetime``, which is the **starting** time of each step."""

    metrics: Optional[SAOEMetrics]
    """Metrics. Only available when done."""

    twap_price: float
    """This price is used to compute price advantage.
    It"s defined as the average price in the period from order"s start time to end time."""

    ticks_index: pd.DatetimeIndex
    """All available ticks for the day (not restricted to order)."""

    ticks_for_order: pd.DatetimeIndex
    """Ticks that is available for trading (sliced by order)."""

    def __init__(
        self,
        order: Order,
        data_dir: Path,
        feature_columns_today: List[str] = [],
        feature_columns_yesterday: List[str] = [],
        data_granularity: int = 1,
        ticks_per_step: int = 30,
        vol_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(initial=order)

        assert ticks_per_step % data_granularity == 0

        self.order = order
        self.data_dir = data_dir
        self.feature_columns_today = feature_columns_today
        self.feature_columns_yesterday = feature_columns_yesterday
        self.ticks_per_step: int = ticks_per_step // data_granularity
        self.vol_threshold = vol_threshold

        self.backtest_data = self.get_backtest_data()
        self.ticks_index = self.backtest_data.get_time_index()

        # Get time index available for trading
        self.ticks_for_order = self._get_ticks_slice(self.order.start_time, self.order.end_time)

        self.cur_time = self.ticks_for_order[0]
        self.cur_step = 0
        # NOTE: astype(float) is necessary in some systems.
        # this will align the precision with `.to_numpy()` in `_split_exec_vol`
        self.twap_price = float(self.backtest_data.get_deal_price().loc[self.ticks_for_order].astype(float).mean())

        self.position = order.amount

        metric_keys = list(SAOEMetrics.__annotations__.keys())  # pylint: disable=no-member
        # NOTE: can empty dataframe contain index?
        self.history_exec = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.history_steps = pd.DataFrame(columns=metric_keys).set_index("datetime")
        self.metrics = None

        self.market_price: Optional[np.ndarray] = None
        self.market_vol: Optional[np.ndarray] = None
        self.market_vol_limit: Optional[np.ndarray] = None

    def get_backtest_data(self) -> BaseIntradayBacktestData:
        try:
            data = load_handler_intraday_processed_data(
                data_dir=self.data_dir,
                stock_id=self.order.stock_id,
                date=pd.Timestamp(self.order.start_time.date()),
                feature_columns_today=self.feature_columns_today,
                feature_columns_yesterday=self.feature_columns_yesterday,
                backtest=True,
                index_only=False,
            )
            return DataframeIntradayBacktestData(data.today)
        except (AttributeError, FileNotFoundError):
            # TODO: For compatibility with older versions of test scripts (tests/rl/test_saoe_simple.py)
            # TODO: In the future, we should modify the data format used by the test script,
            # TODO: and then delete this branch.
            return load_simple_intraday_backtest_data(
                self.data_dir / "backtest",
                self.order.stock_id,
                pd.Timestamp(self.order.start_time.date()),
                "close",
                self.order.direction,
            )

    def step(self, amount: float) -> None:
        """Execute one step or SAOE.

        Parameters
        ----------
        amount
            The amount you wish to deal. The simulator doesn't guarantee all the amount to be successfully dealt.
        """

        assert not self.done()

        self.market_price = self.market_vol = None  # avoid misuse
        exec_vol = self._split_exec_vol(amount)
        assert self.market_price is not None
        assert self.market_vol is not None

        ticks_position = self.position - np.cumsum(exec_vol)

        self.position -= exec_vol.sum()
        if abs(self.position) < 1e-6:
            self.position = 0.0
        if self.position < -EPS or (exec_vol < -EPS).any():
            raise ValueError(f"Execution volume is invalid: {exec_vol} (position = {self.position})")

        # Get time index available for this step
        time_index = self._get_ticks_slice(self.cur_time, self._next_time())

        self.history_exec = self._dataframe_append(
            self.history_exec,
            SAOEMetrics(
                # It should have the same keys with SAOEMetrics,
                # but the values do not necessarily have the annotated type.
                # Some values could be vectorized (e.g., exec_vol).
                stock_id=self.order.stock_id,
                datetime=time_index,
                direction=self.order.direction,
                market_volume=self.market_vol,
                market_price=self.market_price,
                amount=exec_vol,
                inner_amount=exec_vol,
                deal_amount=exec_vol,
                trade_price=self.market_price,
                trade_value=self.market_price * exec_vol,
                position=ticks_position,
                ffr=exec_vol / self.order.amount,
                pa=price_advantage(self.market_price, self.twap_price, self.order.direction),
            ),
        )

        self.history_steps = self._dataframe_append(
            self.history_steps,
            [self._metrics_collect(self.cur_time, self.market_vol, self.market_price, amount, exec_vol)],
        )

        if self.done():
            if self.env is not None:
                self.env.logger.add_any("history_steps", self.history_steps, loglevel=LogLevel.DEBUG)
                self.env.logger.add_any("history_exec", self.history_exec, loglevel=LogLevel.DEBUG)

            self.metrics = self._metrics_collect(
                self.ticks_index[0],  # start time
                self.history_exec["market_volume"],
                self.history_exec["market_price"],
                self.history_steps["amount"].sum(),
                self.history_exec["deal_amount"],
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

    def get_state(self) -> SAOEState:
        return SAOEState(
            order=self.order,
            cur_time=self.cur_time,
            cur_step=self.cur_step,
            position=self.position,
            history_exec=self.history_exec,
            history_steps=self.history_steps,
            metrics=self.metrics,
            backtest_data=self.backtest_data,
            ticks_per_step=self.ticks_per_step,
            ticks_index=self.ticks_index,
            ticks_for_order=self.ticks_for_order,
        )

    def done(self) -> bool:
        return self.position < EPS or self.cur_time >= self.order.end_time

    def _next_time(self) -> pd.Timestamp:
        """The "current time" (``cur_time``) for next step."""
        # Look for next time on time index
        current_loc = self.ticks_index.get_loc(self.cur_time)
        next_loc = current_loc + self.ticks_per_step

        # Calibrate the next location to multiple of ticks_per_step.
        # This is to make sure that:
        # as long as ticks_per_step is a multiple of something, each step won't cross morning and afternoon.
        next_loc = next_loc - next_loc % self.ticks_per_step

        if next_loc < len(self.ticks_index) and self.ticks_index[next_loc] < self.order.end_time:
            return self.ticks_index[next_loc]
        else:
            return self.order.end_time

    def _cur_duration(self) -> pd.Timedelta:
        """The "duration" of this step (step that is about to happen)."""
        return self._next_time() - self.cur_time

    def _split_exec_vol(self, exec_vol_sum: float) -> np.ndarray:
        """
        Split the volume in each step into minutes, considering possible constraints.
        This follows TWAP strategy.
        """
        next_time = self._next_time()

        # get the backtest data for next interval
        self.market_vol = self.backtest_data.get_volume().loc[self.cur_time : next_time - EPS_T].to_numpy()
        self.market_price = self.backtest_data.get_deal_price().loc[self.cur_time : next_time - EPS_T].to_numpy()

        assert self.market_vol is not None and self.market_price is not None

        # split the volume equally into each minute
        exec_vol = np.repeat(exec_vol_sum / len(self.market_price), len(self.market_price))

        # apply the volume threshold
        market_vol_limit = self.vol_threshold * self.market_vol if self.vol_threshold is not None else np.inf
        exec_vol = np.minimum(exec_vol, market_vol_limit)  # type: ignore

        # Complete all the order amount at the last moment.
        if next_time >= self.order.end_time:
            exec_vol[-1] += self.position - exec_vol.sum()
            exec_vol = np.minimum(exec_vol, market_vol_limit)  # type: ignore

        return exec_vol

    def _metrics_collect(
        self,
        datetime: pd.Timestamp,
        market_vol: np.ndarray,
        market_price: np.ndarray,
        amount: float,  # intended to trade such amount
        exec_vol: np.ndarray,
    ) -> SAOEMetrics:
        assert len(market_vol) == len(market_price) == len(exec_vol)

        if np.abs(np.sum(exec_vol)) < EPS:
            exec_avg_price = 0.0
        else:
            exec_avg_price = cast(float, np.average(market_price, weights=exec_vol))  # could be nan
            if hasattr(exec_avg_price, "item"):  # could be numpy scalar
                exec_avg_price = exec_avg_price.item()  # type: ignore

        return SAOEMetrics(
            stock_id=self.order.stock_id,
            datetime=datetime,
            direction=self.order.direction,
            market_volume=market_vol.sum(),
            market_price=market_price.mean(),
            amount=amount,
            inner_amount=exec_vol.sum(),
            deal_amount=exec_vol.sum(),  # in this simulator, there's no other restrictions
            trade_price=exec_avg_price,
            trade_value=float(np.sum(market_price * exec_vol)),
            position=self.position,
            ffr=float(exec_vol.sum() / self.order.amount),
            pa=price_advantage(exec_avg_price, self.twap_price, self.order.direction),
        )

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


def price_advantage(
    exec_price: float_or_ndarray,
    baseline_price: float,
    direction: OrderDir | int,
) -> float_or_ndarray:
    if baseline_price == 0:  # something is wrong with data. Should be nan here
        if isinstance(exec_price, float):
            return 0.0
        else:
            return np.zeros_like(exec_price)
    if direction == OrderDir.BUY:
        res = (1 - exec_price / baseline_price) * 10000
    elif direction == OrderDir.SELL:
        res = (exec_price / baseline_price - 1) * 10000
    else:
        raise ValueError(f"Unexpected order direction: {direction}")
    res_wo_nan: np.ndarray = np.nan_to_num(res, nan=0.0)
    if res_wo_nan.size == 1:
        return res_wo_nan.item()
    else:
        return cast(float_or_ndarray, res_wo_nan)
