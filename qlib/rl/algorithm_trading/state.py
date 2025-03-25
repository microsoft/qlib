# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import typing
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from qlib.backtest.decision import Task
from qlib.typehint import TypedDict

if typing.TYPE_CHECKING:
    from qlib.rl.data.base import BaseIntradayBacktestData


class SAATMetrics(TypedDict):
    """Metrics for SAAT accumulated for a "period".
    It could be accumulated for a day, or a period of time (e.g., 30min), or calculated separately for every minute.

    Warnings
    --------
    The type hints are for single elements. In lots of times, they can be vectorized.
    For example, ``market_volume`` could be a list of float (or ndarray) rather tahn a single float.
    """

    stock_id: str
    """Stock ID of this record."""
    datetime: pd.Timestamp | pd.DatetimeIndex
    """Datetime of this record (this is index in the dataframe)."""
    direction: int
    """Direction to support reuse order. 0 for sell, 1 for buy, 2 for algorithm trading."""

    # Market information.
    market_volume: np.ndarray | float
    """(total) market volume traded in the period."""
    market_price: np.ndarray | float
    """Deal price. If it's a period of time, this is the average market deal price."""

    # Strategy records.
    action: np.ndarray | float
    """Next step action."""
    trade_price: np.ndarray | float
    """The average deal price for this strategy."""
    trading_value: np.ndarray | float
    """Total worth of trading. In the simple simulation, trade_value = deal_amount * price."""
    position: np.ndarray | float
    """Position after this step."""
    cash: np.ndarray | float
    """Cash after this step."""
    total_cash: np.ndarray | float
    """Total cash used for trading."""

    # Accumulated metrics
    ret: np.ndarray | float
    """Return."""
    swap_value: np.ndarray | int
    """Swap value for calculating transaction fee."""


class SAATState(NamedTuple):
    """Data structure holding a state for SAAT simulator."""

    task: Task
    """The stock we are dealing with."""
    cur_time: pd.Timestamp
    """Current time, e.g., 9:30."""
    cur_step: int
    """Current step, e.g., 0."""
    cash: float
    """Current remaining cash can be used."""
    position: float
    """Current position."""
    history_exec: pd.DataFrame
    """See :attr:`SingleAssetAlgorithmTrading.history_exec`."""
    history_steps: pd.DataFrame
    """See :attr:`SingleAssetAlgorithmTrading.history_steps`."""
    metrics: Optional[SAATMetrics]
    """Daily metric, only available when the trading is in "done" state."""
    backtest_data: BaseIntradayBacktestData
    """Backtest data is included in the state.
    Actually, only the time index of this data is needed, at this moment.
    I include the full data so that algorithms (e.g., VWAP) that relies on the raw data can be implemented.
    Interpreter can use this as they wish, but they should be careful not to leak future data.
    """
    ticks_per_step: int
    """How many ticks for each step."""
    ticks_index: pd.DatetimeIndex
    """Trading ticks in all day, NOT sliced by task (defined in data). e.g., [9:30, 9:31, ..., 14:59]."""
    ticks_for_trading: pd.DatetimeIndex
    """Trading ticks sliced by trading, e.g., [9:45, 9:46, ..., 14:44]."""
