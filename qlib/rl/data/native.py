# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from pathlib import Path
from typing import cast, List

import cachetools
import pandas as pd
import pickle
import os

from qlib.backtest import Exchange, Order
from qlib.backtest.decision import TradeRange, TradeRangeByTime
from qlib.constant import EPS_T
from .base import BaseIntradayBacktestData, BaseIntradayProcessedData, ProcessedDataProvider


def get_ticks_slice(
    ticks_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_end: bool = False,
) -> pd.DatetimeIndex:
    if not include_end:
        end = end - EPS_T
    return ticks_index[ticks_index.slice_indexer(start, end)]


class IntradayBacktestData(BaseIntradayBacktestData):
    """Backtest data for Qlib simulator"""

    def __init__(
        self,
        order: Order,
        exchange: Exchange,
        ticks_index: pd.DatetimeIndex,
        ticks_for_order: pd.DatetimeIndex,
    ) -> None:
        self._order = order
        self._exchange = exchange
        self._start_time = ticks_for_order[0]
        self._end_time = ticks_for_order[-1]
        self.ticks_index = ticks_index
        self.ticks_for_order = ticks_for_order

        self._deal_price = cast(
            pd.Series,
            self._exchange.get_deal_price(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                direction=self._order.direction,
                method=None,
            ),
        )
        self._volume = cast(
            pd.Series,
            self._exchange.get_volume(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                method=None,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"Order: {self._order}, Exchange: {self._exchange}, "
            f"Start time: {self._start_time}, End time: {self._end_time}"
        )

    def __len__(self) -> int:
        return len(self._deal_price)

    def get_deal_price(self) -> pd.Series:
        return self._deal_price

    def get_volume(self) -> pd.Series:
        return self._volume

    def get_time_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([e[1] for e in list(self._exchange.quote_df.index)])


class DataframeIntradayBacktestData(BaseIntradayBacktestData):
    """Backtest data from dataframe"""

    def __init__(self, df: pd.DataFrame, price_column: str = "$close0", volume_column: str = "$volume0") -> None:
        self.df = df
        self.price_column = price_column
        self.volume_column = volume_column

    def __repr__(self) -> str:
        with pd.option_context("memory_usage", False, "display.max_info_columns", 1, "display.large_repr", "info"):
            return f"{self.__class__.__name__}({self.df})"

    def __len__(self) -> int:
        return len(self.df)

    def get_deal_price(self) -> pd.Series:
        return self.df[self.price_column]

    def get_volume(self) -> pd.Series:
        return self.df[self.volume_column]

    def get_time_index(self) -> pd.DatetimeIndex:
        return cast(pd.DatetimeIndex, self.df.index)


@cachetools.cached(  # type: ignore
    cache=cachetools.LRUCache(100),
    key=lambda order, _, __: order.key_by_day,
)
def load_backtest_data(
    order: Order,
    trade_exchange: Exchange,
    trade_range: TradeRange,
) -> IntradayBacktestData:
    ticks_index = pd.DatetimeIndex(trade_exchange.quote_df.reset_index()["datetime"])
    ticks_index = ticks_index[order.start_time <= ticks_index]
    ticks_index = ticks_index[ticks_index <= order.end_time]

    if isinstance(trade_range, TradeRangeByTime):
        ticks_for_order = get_ticks_slice(
            ticks_index,
            trade_range.start_time,
            trade_range.end_time,
            include_end=True,
        )
    else:
        ticks_for_order = None  # FIXME: implement this logic

    backtest_data = IntradayBacktestData(
        order=order,
        exchange=trade_exchange,
        ticks_index=ticks_index,
        ticks_for_order=ticks_for_order,
    )
    return backtest_data


class HandlerIntradayProcessedData(BaseIntradayProcessedData):
    """Subclass of IntradayProcessedData. Used to handle handler (bin format) style data."""

    def __init__(
        self,
        data_dir: Path,
        stock_id: str,
        date: pd.Timestamp,
        feature_columns_today: List[str],
        feature_columns_yesterday: List[str],
        backtest: bool = False,
        index_only: bool = False,
    ) -> None:
        def _drop_stock_id(df: pd.DataFrame) -> pd.DataFrame:
            df = df.reset_index()
            if "instrument" in df.columns:
                df = df.drop(columns=["instrument"])
            return df.set_index(["datetime"])

        path = os.path.join(data_dir, "backtest" if backtest else "feature", f"{stock_id}.pkl")
        start_time, end_time = date.replace(hour=0, minute=0, second=0), date.replace(hour=23, minute=59, second=59)
        with open(path, "rb") as fstream:
            dataset = pickle.load(fstream)
        data = dataset.handler.fetch(pd.IndexSlice[stock_id, start_time:end_time], level=None)

        if index_only:
            self.today = _drop_stock_id(data[[]])
            self.yesterday = _drop_stock_id(data[[]])
        else:
            self.today = _drop_stock_id(data[feature_columns_today])
            self.yesterday = _drop_stock_id(data[feature_columns_yesterday])

    def __repr__(self) -> str:
        with pd.option_context("memory_usage", False, "display.max_info_columns", 1, "display.large_repr", "info"):
            return f"{self.__class__.__name__}({self.today}, {self.yesterday})"


@cachetools.cached(  # type: ignore
    cache=cachetools.LRUCache(100),  # 100 * 50K = 5MB
    key=lambda data_dir, stock_id, date, feature_columns_today, feature_columns_yesterday, backtest, index_only: (
        stock_id,
        date,
        backtest,
        index_only,
    ),
)
def load_handler_intraday_processed_data(
    data_dir: Path,
    stock_id: str,
    date: pd.Timestamp,
    feature_columns_today: List[str],
    feature_columns_yesterday: List[str],
    backtest: bool = False,
    index_only: bool = False,
) -> HandlerIntradayProcessedData:
    return HandlerIntradayProcessedData(
        data_dir, stock_id, date, feature_columns_today, feature_columns_yesterday, backtest, index_only
    )


class HandlerProcessedDataProvider(ProcessedDataProvider):
    def __init__(
        self,
        data_dir: str,
        feature_columns_today: List[str],
        feature_columns_yesterday: List[str],
        backtest: bool = False,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.feature_columns_today = feature_columns_today
        self.feature_columns_yesterday = feature_columns_yesterday
        self.backtest = backtest

    def get_data(
        self,
        stock_id: str,
        date: pd.Timestamp,
        feature_dim: int,
        time_index: pd.Index,
    ) -> BaseIntradayProcessedData:
        return load_handler_intraday_processed_data(
            self.data_dir,
            stock_id,
            date,
            self.feature_columns_today,
            self.feature_columns_yesterday,
            backtest=self.backtest,
            index_only=False,
        )
