# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import typing
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from qlib.backtest import Order
from qlib.typehint import TypedDict

if typing.TYPE_CHECKING:
    from qlib.rl.data.base import BaseIntradayBacktestData


class SAOEMetrics(TypedDict):
    """Metrics for SAOE accumulated for a "period".
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
    """Direction of the order. 0 for sell, 1 for buy."""

    # Market information.
    market_volume: np.ndarray | float
    """(total) market volume traded in the period."""
    market_price: np.ndarray | float
    """Deal price. If it's a period of time, this is the average market deal price."""

    # Strategy records.

    amount: np.ndarray | float
    """Total amount (volume) strategy intends to trade."""
    inner_amount: np.ndarray | float
    """Total amount that the lower-level strategy intends to trade
    (might be larger than amount, e.g., to ensure ffr)."""

    deal_amount: np.ndarray | float
    """Amount that successfully takes effect (must be less than inner_amount)."""
    trade_price: np.ndarray | float
    """The average deal price for this strategy."""
    trade_value: np.ndarray | float
    """Total worth of trading. In the simple simulation, trade_value = deal_amount * price."""
    position: np.ndarray | float
    """Position left after this "period"."""

    # Accumulated metrics

    ffr: np.ndarray | float
    """Completed how much percent of the daily order."""

    pa: np.ndarray | float
    """Price advantage compared to baseline (i.e., trade with baseline market price).
    The baseline is trade price when using TWAP strategy to execute this order.
    Please note that there could be data leak here).
    Unit is BP (basis point, 1/10000)."""


class SAOEState(NamedTuple):
    """Data structure holding a state for SAOE simulator."""

    order: Order
    """The order we are dealing with."""
    cur_time: pd.Timestamp
    """Current time, e.g., 9:30."""
    cur_step: int
    """Current step, e.g., 0."""
    position: float
    """Current remaining volume to execute."""
    history_exec: pd.DataFrame
    """See :attr:`SingleAssetOrderExecution.history_exec`."""
    history_steps: pd.DataFrame
    """See :attr:`SingleAssetOrderExecution.history_steps`."""

    metrics: Optional[SAOEMetrics]
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
    """Trading ticks in all day, NOT sliced by order (defined in data). e.g., [9:30, 9:31, ..., 14:59]."""
    ticks_for_order: pd.DatetimeIndex
    """Trading ticks sliced by order, e.g., [9:45, 9:46, ..., 14:44]."""
