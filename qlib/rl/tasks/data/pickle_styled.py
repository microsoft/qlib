# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module contains utilities to read financial data from pickle-styled files.

This is the format used in `OPD paper <https://seqml.github.io/opd/>`__. NOT the standard data format in qlib.

The data here are all wrapped with ``@lru_cache``, which saves the expensive IO cost to repetitively read the data.
We also encourage users to use ``get_xxx_yyy`` rather than ``XxxYyy`` (although they are the same thing),
because ``get_xxx_yyy`` is cache-optimized.
"""

from functools import lru_cache
from typing import Literal, List
from pathlib import Path

import cachetools
import numpy as np
import pandas as pd
from cachetools.keys import hashkey

from qlib.backtest.decision import OrderDir


DealPriceType = Literal['bid_or_ask', 'bid_or_ask_fill', 'close']
"""Several ad-hoc deal price.
``bid_or_ask``: If sell, use column ``$bid0``; if buy, use column ``$ask0``.
``bid_or_ask_fill``: Based on ``bid_or_ask``. If price is 0, use another price (``$ask0`` / ``$bid0``) instead.
``close``: Use close price (``$close0``) as deal price.
"""


def _infer_processed_data_column_names(shape):
    if shape == 16:
        return [
            "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume",
            "$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5"
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
        raise FileNotFoundError(f'No file starting with "{filename_without_suffix}" found')
    if len(paths) > 1:
        raise ValueError(f'Multiple paths are found with prefix "{filename_without_suffix}": {paths}')
    return paths[0]


@lru_cache(maxsize=10)  # 10 * 40M = 400MB
def _read_pickle(filename_without_suffix: Path) -> pd.DataFrame:
    return pd.read_pickle(_find_pickle(filename_without_suffix))


class IntradayBacktestData:
    """Raw market data that is often used in backtesting (thus called BacktestData)."""

    def __init__(self, data_dir: Path, stock_id: str, date: pd.Timestamp, deal_price: DealPriceType):
        backtest = _read_pickle(data_dir / stock_id)
        backtest = backtest.loc[pd.IndexSlice[stock_id, :, date]].droplevel([0, 2])

        self.data: pd.DataFrame = backtest
        self.deal_price_type: DealPriceType = deal_price

    def __repr__(self):
        return f'{self.__class__.__name__}({self.data})'

    def get_deal_price(self, order_dir: int) -> pd.Series:
        """Return a pandas series that can be indexed with time.
        See :attribute:`DealPriceType` for details."""
        if self.deal_price_type in ('bid_or_ask', 'bid_or_ask_fill'):
            if order_dir == OrderDir.SELL:
                col = '$bid0'
            else:               # BUY
                col = '$ask0'
        elif self.deal_price_type == 'close':
            col = '$close0'
        price = self.data[col]

        if self.deal_price_type == 'bid_or_ask_fill':
            if order_dir == OrderDir.SELL:
                fill_col = '$ask0'
            else:
                fill_col = '$bid0'
            price = price.replace(0, np.nan).fillna(self.data[fill_col])

        return price

    def get_volume(self) -> pd.Series:
        """Return a volume series that can be indexed with time."""
        return self.data['$volume0']

    def get_time_index(self) -> pd.DatetimeIndex:
        return self.data.index


class IntradayProcessedData:
    """Processed market data after data cleanup and feature engineering.

    It contains both processed data for "today" and "yesterday", as some algorithms
    might use the market information of the previous day to assist decision making.
    """

    proc_today: pd.DataFrame
    """Processed data for "today".
    Number of records must be ``time_length``, and columns must be ``feature_dim``."""

    proc_yesterday: pd.DataFrame
    """Processed data for "yesterday".
    Number of records must be ``time_length``, and columns must be ``feature_dim``."""

    def __init__(
        self, data_dir: Path, stock_id: str, date: pd.Timestamp,
        feature_dim: int, time_index: pd.Index
    ):
        proc = _read_pickle(data_dir / stock_id)
        # We have to infer the names here because,
        # unfortunately they are not included in the original data.
        cnames = _infer_processed_data_column_names(feature_dim)

        time_length: int = len(time_index)

        try:
            # new data format
            proc = proc.loc[pd.IndexSlice[stock_id, :, date]]
            assert len(proc) == time_length and len(proc.columns) == feature_dim * 2
            proc_today = proc[cnames]
            proc_yesterday = proc[[f'{c}_1' for c in cnames]].rename(columns=lambda c: c[:-2])
        except (IndexError, KeyError):
            # legacy data
            proc = proc.loc[pd.IndexSlice[stock_id, date]]
            assert time_length * feature_dim * 2 == len(proc)
            proc_today = proc.to_numpy()[: time_length * feature_dim].reshape((time_length, feature_dim))
            proc_yesterday = proc.to_numpy()[time_length * feature_dim:].reshape((time_length, feature_dim))
            proc_today = pd.DataFrame(proc_today, index=time_index, columns=cnames)
            proc_yesterday = pd.DataFrame(proc_yesterday, index=time_index, columns=cnames)

        self.today: pd.DataFrame = proc_today
        self.yesterday: pd.DataFrame = proc_yesterday
        assert len(self.today.columns) == len(self.yesterday.columns) == feature_dim
        assert len(self.today) == len(self.yesterday) == time_length

    def __repr__(self):
        return f'{self.__class__.__name__}({self.today}, {self.yesterday})'


@lru_cache(maxsize=100)  # 100 * 50K = 5MB
def get_intraday_backtest_data(
    data_dir: Path, stock_id: str, date: pd.Timestamp, deal_price: DealPriceType
) -> IntradayBacktestData:
    return IntradayBacktestData(data_dir, stock_id, date, deal_price)


@cachetools.cached(
    cache=cachetools.LRUCache(100),  # 100 * 50K = 5MB
    key=lambda data_dir, stock_id, date, _, __: hashkey(data_dir, stock_id, date)
)
def get_intraday_processed_data(
    data_dir: Path, stock_id: str, date: pd.Timestamp,
    feature_dim: int, time_index: pd.Index
) -> IntradayProcessedData:
    return IntradayProcessedData(data_dir, stock_id, date, feature_dim, time_index)
