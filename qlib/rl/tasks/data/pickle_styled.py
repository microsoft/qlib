# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module contains utilities to read financial data from pickle-styled files.

This is the format used in `OPD paper <https://seqml.github.io/opd/>`__. NOT the standard data format in qlib.
"""

from typing import Literal, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

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
    for suffix in suffix_list:
        path = filename_without_suffix.parent / (filename_without_suffix.name + suffix)
        if path.exists():
            return path
    raise FileNotFoundError(f'No file starting with "{filename_without_suffix}" found')


def get_deal_price(data: pd.DataFrame, deal_price: DealPriceType, order_dir: int) -> pd.Series:
    """See :attribute:`DealPriceType` for details."""
    if deal_price in ('bid_or_ask', 'bid_or_ask_fill'):
        if order_dir == OrderDir.SELL:
            col = '$bid0'
        else:               # BUY
            col = '$ask0'
    elif deal_price == 'close':
        col = '$close0'
    price = data[col]

    if deal_price == 'bid_or_ask_fill':
        if order_dir == OrderDir.SELL:
            fill_col = '$ask0'
        else:
            fill_col = '$bid0'
        price = price.replace(0, np.nan).fillna(data[fill_col])

    return price


def get_intraday_backtest_data(data_dir: Path, stock_id: str, date: pd.Timestamp) -> pd.DataFrame:
    backtest = pd.read_pickle(_find_pickle(data_dir / stock_id))
    backtest = backtest.loc[pd.IndexSlice[stock_id, :, date]].droplevel([0, 2])
    return backtest


def get_intraday_processed_data(
    data_dir: Path, stock_id: str, date: pd.Timestamp,
    time_length: int, feature_dim: int, time_index: pd.Index
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the processed data for today and yesterday. Returns a tuple."""
    proc = pd.read_pickle(_find_pickle(data_dir / stock_id))
    cnames = _infer_processed_data_column_names(feature_dim)
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

    return proc_today, proc_yesterday
