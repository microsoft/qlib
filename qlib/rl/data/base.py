# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from abc import abstractmethod

import pandas as pd


class BaseIntradayBacktestData:
    """
    Raw market data that is often used in backtesting (thus called BacktestData).

    Base class for all types of backtest data. Currently, each type of simulator has its corresponding backtest
    data type.
    """

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_deal_price(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_volume(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_time_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError


class BaseIntradayProcessedData:
    """Processed market data after data cleanup and feature engineering.

    It contains both processed data for "today" and "yesterday", as some algorithms
    might use the market information of the previous day to assist decision making.
    """

    today: pd.DataFrame
    """Processed data for "today".
    Number of records must be ``time_length``, and columns must be ``feature_dim``."""

    yesterday: pd.DataFrame
    """Processed data for "yesterday".
    Number of records must be ``time_length``, and columns must be ``feature_dim``."""


class ProcessedDataProvider:
    """Provider of processed data"""

    def get_data(
        self,
        stock_id: str,
        date: pd.Timestamp,
        feature_dim: int,
        time_index: pd.Index,
    ) -> BaseIntradayProcessedData:
        raise NotImplementedError
