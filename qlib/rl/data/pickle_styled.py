# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module contains utilities to read financial data from pickle-styled files.

This is the format used in `OPD paper <https://seqml.github.io/opd/>`__. NOT the standard data format in qlib.

The data here are all wrapped with ``@lru_cache``, which saves the expensive IO cost to repetitively read the data.
We also encourage users to use ``get_xxx_yyy`` rather than ``XxxYyy`` (although they are the same thing),
because ``get_xxx_yyy`` is cache-optimized.

Note that these pickle files are dumped with Python 3.8. Python lower than 3.7 might not be able to load them.
See `PEP 574 <https://peps.python.org/pep-0574/>`__ for details.

This file shows resemblence to qlib.backtest.high_performance_ds. We might merge those two in future.
"""

# TODO: merge with qlib/backtest/high_performance_ds.py

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, cast

import cachetools
import numpy as np
import pandas as pd
from cachetools.keys import hashkey

from qlib.backtest.decision import Order, OrderDir
from qlib.rl.data.base import BaseIntradayBacktestData, BaseIntradayProcessedData, ProcessedDataProvider
from qlib.typehint import Literal

DealPriceType = Literal["bid_or_ask", "bid_or_ask_fill", "close"]
"""Several ad-hoc deal price.
``bid_or_ask``: If sell, use column ``$bid0``; if buy, use column ``$ask0``.
``bid_or_ask_fill``: Based on ``bid_or_ask``. If price is 0, use another price (``$ask0`` / ``$bid0``) instead.
``close``: Use close price (``$close0``) as deal price.
"""


def _infer_processed_data_column_names(shape: int) -> List[str]:
    if shape == 16:
        return [
            "$open",
            "$high",
            "$low",
            "$close",
            "$vwap",
            "$bid",
            "$ask",
            "$volume",
            "$bidV",
            "$bidV1",
            "$bidV3",
            "$bidV5",
            "$askV",
            "$askV1",
            "$askV3",
            "$askV5",
        ]
    if shape == 6:
        return ["$high", "$low", "$open", "$close", "$vwap", "$volume"]
    elif shape == 5:
        return ["$high", "$low", "$open", "$close", "$volume"]
    raise ValueError(f"Unrecognized data shape: {shape}")


def _find_pickle(filename_without_suffix: Path) -> Path:
    suffix_list = [".pkl", ".pkl.backtest"]
    paths: List[Path] = []
    for suffix in suffix_list:
        path = filename_without_suffix.parent / (filename_without_suffix.name + suffix)
        if path.exists():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No file starting with '{filename_without_suffix}' found")
    if len(paths) > 1:
        raise ValueError(f"Multiple paths are found with prefix '{filename_without_suffix}': {paths}")
    return paths[0]


@lru_cache(maxsize=10)  # 10 * 40M = 400MB
def _read_pickle(filename_without_suffix: Path) -> pd.DataFrame:
    df = pd.read_pickle(_find_pickle(filename_without_suffix))
    index_cols = df.index.names

    df = df.reset_index()
    for date_col_name in ["date", "datetime"]:
        if date_col_name in df:
            df[date_col_name] = pd.to_datetime(df[date_col_name])
    df = df.set_index(index_cols)

    return df


class SimpleIntradayBacktestData(BaseIntradayBacktestData):
    """Backtest data for simple simulator"""

    def __init__(
        self,
        data_dir: Path | str,
        stock_id: str,
        date: pd.Timestamp,
        deal_price: DealPriceType = "close",
        order_dir: int | None = None,
    ) -> None:
        super(SimpleIntradayBacktestData, self).__init__()

        backtest = _read_pickle((data_dir if isinstance(data_dir, Path) else Path(data_dir)) / stock_id)
        backtest = backtest.loc[pd.IndexSlice[stock_id, :, date]]

        # No longer need for pandas >= 1.4
        # backtest = backtest.droplevel([0, 2])

        self.data: pd.DataFrame = backtest
        self.deal_price_type: DealPriceType = deal_price
        self.order_dir = order_dir

    def __repr__(self) -> str:
        with pd.option_context("memory_usage", False, "display.max_info_columns", 1, "display.large_repr", "info"):
            return f"{self.__class__.__name__}({self.data})"

    def __len__(self) -> int:
        return len(self.data)

    def get_deal_price(self) -> pd.Series:
        """Return a pandas series that can be indexed with time.
        See :attribute:`DealPriceType` for details."""
        if self.deal_price_type in ("bid_or_ask", "bid_or_ask_fill"):
            if self.order_dir is None:
                raise ValueError("Order direction cannot be none when deal_price_type is not close.")
            if self.order_dir == OrderDir.SELL:
                col = "$bid0"
            else:  # BUY
                col = "$ask0"
        elif self.deal_price_type == "close":
            col = "$close0"
        else:
            raise ValueError(f"Unsupported deal_price_type: {self.deal_price_type}")
        price = self.data[col]

        if self.deal_price_type == "bid_or_ask_fill":
            if self.order_dir == OrderDir.SELL:
                fill_col = "$ask0"
            else:
                fill_col = "$bid0"
            price = price.replace(0, np.nan).fillna(self.data[fill_col])

        return price

    def get_volume(self) -> pd.Series:
        """Return a volume series that can be indexed with time."""
        return self.data["$volume0"]

    def get_time_index(self) -> pd.DatetimeIndex:
        return cast(pd.DatetimeIndex, self.data.index)


class PickleIntradayProcessedData(BaseIntradayProcessedData):
    """Subclass of IntradayProcessedData. Used to handle pickle-styled data."""

    def __init__(
        self,
        data_dir: Path | str,
        stock_id: str,
        date: pd.Timestamp,
        feature_dim: int,
        time_index: pd.Index,
    ) -> None:
        proc = _read_pickle((data_dir if isinstance(data_dir, Path) else Path(data_dir)) / stock_id)

        # We have to infer the names here because,
        # unfortunately they are not included in the original data.
        cnames = _infer_processed_data_column_names(feature_dim)

        time_length: int = len(time_index)

        try:
            # new data format
            proc = proc.loc[pd.IndexSlice[stock_id, :, date]]
            assert len(proc) == time_length and len(proc.columns) == feature_dim * 2
            proc_today = proc[cnames]
            proc_yesterday = proc[[f"{c}_1" for c in cnames]].rename(columns=lambda c: c[:-2])
        except (IndexError, KeyError):
            # legacy data
            proc = proc.loc[pd.IndexSlice[stock_id, date]]
            assert time_length * feature_dim * 2 == len(proc)
            proc_today = proc.to_numpy()[: time_length * feature_dim].reshape((time_length, feature_dim))
            proc_yesterday = proc.to_numpy()[time_length * feature_dim :].reshape((time_length, feature_dim))
            proc_today = pd.DataFrame(proc_today, index=time_index, columns=cnames)
            proc_yesterday = pd.DataFrame(proc_yesterday, index=time_index, columns=cnames)

        self.today: pd.DataFrame = proc_today
        self.yesterday: pd.DataFrame = proc_yesterday
        assert len(self.today.columns) == len(self.yesterday.columns) == feature_dim
        assert len(self.today) == len(self.yesterday) == time_length

    def __repr__(self) -> str:
        with pd.option_context("memory_usage", False, "display.max_info_columns", 1, "display.large_repr", "info"):
            return f"{self.__class__.__name__}({self.today}, {self.yesterday})"


@lru_cache(maxsize=100)  # 100 * 50K = 5MB
def load_simple_intraday_backtest_data(
    data_dir: Path,
    stock_id: str,
    date: pd.Timestamp,
    deal_price: DealPriceType = "close",
    order_dir: int | None = None,
) -> SimpleIntradayBacktestData:
    return SimpleIntradayBacktestData(data_dir, stock_id, date, deal_price, order_dir)


@cachetools.cached(  # type: ignore
    cache=cachetools.LRUCache(100),  # 100 * 50K = 5MB
    key=lambda data_dir, stock_id, date, feature_dim, time_index: hashkey(data_dir, stock_id, date),
)
def load_pickle_intraday_processed_data(
    data_dir: Path,
    stock_id: str,
    date: pd.Timestamp,
    feature_dim: int,
    time_index: pd.Index,
) -> BaseIntradayProcessedData:
    return PickleIntradayProcessedData(data_dir, stock_id, date, feature_dim, time_index)


class PickleProcessedDataProvider(ProcessedDataProvider):
    def __init__(self, data_dir: Path) -> None:
        super().__init__()

        self._data_dir = data_dir

    def get_data(
        self,
        stock_id: str,
        date: pd.Timestamp,
        feature_dim: int,
        time_index: pd.Index,
    ) -> BaseIntradayProcessedData:
        return load_pickle_intraday_processed_data(
            data_dir=self._data_dir,
            stock_id=stock_id,
            date=date,
            feature_dim=feature_dim,
            time_index=time_index,
        )


def load_orders(
    order_path: Path,
    start_time: pd.Timestamp = None,
    end_time: pd.Timestamp = None,
) -> Sequence[Order]:
    """Load orders, and set start time and end time for the orders."""

    start_time = start_time or pd.Timestamp("0:00:00")
    end_time = end_time or pd.Timestamp("23:59:59")

    if order_path.is_file():
        order_df = pd.read_pickle(order_path)
    else:
        order_df = []
        for file in order_path.iterdir():
            order_data = pd.read_pickle(file)
            order_df.append(order_data)
        order_df = pd.concat(order_df)

    order_df = order_df.reset_index()

    # Legacy-style orders have "date" instead of "datetime"
    if "date" in order_df.columns:
        order_df = order_df.rename(columns={"date": "datetime"})

    # Sometimes "date" are str rather than Timestamp
    order_df["datetime"] = pd.to_datetime(order_df["datetime"])

    orders: List[Order] = []

    for _, row in order_df.iterrows():
        # filter out orders with amount == 0
        if row["amount"] <= 0:
            continue
        orders.append(
            Order(
                row["instrument"],
                row["amount"],
                OrderDir(int(row["order_type"])),
                row["datetime"].replace(hour=start_time.hour, minute=start_time.minute, second=start_time.second),
                row["datetime"].replace(hour=end_time.hour, minute=end_time.minute, second=end_time.second),
            ),
        )

    return orders
