from pathlib import Path
from typing import Literal, NamedTuple, Optional

import numpy as np
import pandas as pd

from qlib.backtest import Order

from .base import Simulator
from .data.pickle_styled import get_intraday_backtest_data, get_deal_price, DealPriceType


class SingleAssetOrderExecutionState(NamedTuple):
    """
    Base class for episodic states.
    """

    order: Order                # the order we are dealing with
    cur_time: pd.Timestamp      # current time, e.g., 9:30
    cur_time_idx: int           # current time index, e.g., in 0-239
    position: float             # remaining volume to execute


class SingleAssetOrderExecution(Simulator[Order, SingleAssetOrderExecutionState]):
    """Single-asset order execution (SAOE) simulator.

    Parameters
    ----------
    initial
        The seed to start an SAOE simulator is an order.
    time_per_step
        Elapsed time per step. Unit is fixed to minute for first iteration.
    backtest_data_dir
        Path to load backtest data
    vol_threshold
        Maximum execution volume (divided by market execution volume).
    """

    def __init__(self, order: Order, backtest_data_dir: Path,
                 time_per_step: str = '30min',
                 deal_price_type: DealPriceType = 'close',
                 vol_threshold: Optional[float] = None) -> None:
        self.order = order
        self.time_per_step = pd.Timedelta(time_per_step)
        self.deal_price_type = deal_price_type
        self.vol_threshold = vol_threshold
        self.cur_time = order.start_time
        self.backtest_data_dir = backtest_data_dir
        self.backtest_data = get_intraday_backtest_data(
            self.backtest_data_dir,
            order.stock_id,
            pd.Timestamp(order.start_time.date)
        )

        self.position = order.amount

    def step(self, amount: float) -> None:
        """Execute one step or SAOE.

        Parameters
        ----------
        amount : float
            The amount you wish to deal. Not the final successfully dealed amount though.
        """

        l, r = self.next_interval()
        self.last_interval = (l, r)
        assert 0 <= l < r
        self.last_step_duration = len(exec_vol)
        self.position -= exec_vol.sum()
        assert self.position > -EPSILON and (exec_vol > -EPSILON).all(), \
            f'Execution volume is invalid: {exec_vol} (position = {self.position})'
        self.position_history[self.cur_step + 1] = self.position
        self.cur_time += self.last_step_duration
        self.cur_step += 1
        if self.cur_step == self.num_step:
            assert self.cur_time == self.end_time
        if self.exec_vol is None:
            self.exec_vol = exec_vol
        else:
            self.exec_vol = np.concatenate((self.exec_vol, exec_vol))

        self.done = self.position < EPSILON or self.cur_step == self.num_step
        if self.on_step_end is not None:
            self.on_step_end(l, r, self)
        if self.done:
            self.update_stats()
            if self.on_step_end is not None:
                self.on_episode_end(self)

        raise NotImplementedError()

    def get_state(self) -> SingleAssetOrderExecutionState:
        return SingleAssetOrderExecutionState(
            order=self.order,
            cur_time=self.cur_time,
        )

    def done(self) -> bool:
        raise NotImplementedError()

    def _next_time(self) -> pd.Timestamp:
        """The "current time" (``cur_time``) for next step."""
        return self.cur_time + self._cur_duration()

    def _cur_duration(self) -> pd.Timedelta:
        """The "duration" of this step (step that is about to happen)."""
        return min(self.order.end_time, self.cur_time + self.time_per_step)

    def _split_exec_vol(self, exec_vol_sum: float) -> np.ndarray:
        """
        Split the volume in each step into minutes, considering possible constraints.
        This follows TWAP strategy.
        """
        next_time = self._next_time()
        ONE_SEC = pd.Timedelta('1s')  # use 1 second to exclude the right interval point

        # get the backtest data for next interval
        backtest_interval = self.backtest_data.loc[self.cur_time:next_time - ONE_SEC]
        market_volume = backtest_interval['$volume0'].to_numpy()
        market_price = get_deal_price(backtest_interval, self.deal_price_type, self.order).to_numpy()

        # split the volume equally into each minute
        exec_vol = np.repeat(exec_vol_sum / len(backtest_interval), len(backtest_interval))

        # apply the volume threshold
        vol_limit = self.vol_threshold * market_volume if self.vol_threshold is not None else np.inf
        exec_vol = np.minimum(exec_vol, vol_limit)

        return exec_vol

    def _assure_ffr_100_percent(self, exec_vol: np.ndarray) -> np.ndarray:
        """Complete all the order amount at the last moment."""
        if self.cur_time + duration == self.end_time:
            exec_vol[-1] += self.position - exec_vol.sum()
            exec_vol = np.minimum(exec_vol, vol_limit)
        return exec_vol
