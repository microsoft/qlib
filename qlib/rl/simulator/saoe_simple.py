from pathlib import Path
from typing import Literal, NamedTuple, Optional, List

import numpy as np
import pandas as pd

from qlib.backtest import Order

from .base import Simulator
from .data.pickle_styled import get_intraday_backtest_data, get_deal_price, DealPriceType


EPSILON = 1e-7

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

    Attributes
    ----------
    exec_history
        All execution volumes at every possible time slot.
    position_history
        Positions left at each step. The position before first step is also recorded.
    """

    exec_history: Optional[np.ndarray]
    position_history: List[float]

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

        self.exec_history = None
        self.position_history = [self.position]

        self.market_price: Optional[np.ndarray] = None
        self.market_vol: Optional[np.ndarray] = None
        self.market_vol_limit: Optional[np.ndarray] = None

    def step(self, amount: float) -> None:
        """Execute one step or SAOE.

        Parameters
        ----------
        amount : float
            The amount you wish to deal. Not the final successfully dealed amount though.
        """

        exec_vol = self._split_exec_vol(amount)

        self.position -= exec_vol.sum()
        if self.position < -EPSILON and not (exec_vol < -EPSILON).any():
            raise ValueError(f'Execution volume is invalid: {exec_vol} (position = {self.position})')
        self.position_history.append(self.position)
        self.cur_time = self._next_time()

        if self.exec_history is None:
            self.exec_history = exec_vol
        else:
            self.exec_vol = np.concatenate((self.exec_vol, exec_vol))

        raise NotImplementedError()

    def get_state(self) -> SingleAssetOrderExecutionState:
        return SingleAssetOrderExecutionState(
            order=self.order,
            cur_time=self.cur_time,
        )

    def done(self) -> bool:
        return self.position < EPSILON or self.cur_time >= self.end_time

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
        self.market_price = get_deal_price(backtest_interval, self.deal_price_type, self.order).to_numpy()

        # split the volume equally into each minute
        exec_vol = np.repeat(exec_vol_sum / len(backtest_interval), len(backtest_interval))

        # apply the volume threshold
        self.market_vol_limit = self.vol_threshold * self.market_vol if self.vol_threshold is not None else np.inf
        exec_vol = np.minimum(exec_vol, self.market_vol_limit)

        return exec_vol

    def _assure_ffr_100_percent(self, exec_vol: np.ndarray) -> np.ndarray:
        """Complete all the order amount at the last moment."""
        if self._next_time() == self.end_time:
            exec_vol[-1] += self.position - exec_vol.sum()
            exec_vol = np.minimum(exec_vol, self.market_vol_limit)
        return exec_vol
