# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import abc
from re import I
from typing import List
from tqdm import tqdm
import sys
import copy
import time
import datetime
import importlib
import baostock as bs
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import Iterable

import fire
import requests
import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker
from dateutil.tz import tzlocal

from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.yahoo.collector import YahooNormalizeCN, YahooNormalize1minOffline, YahooNormalize1min
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
    get_us_stock_symbols,
    get_in_stock_symbols,
    get_br_stock_symbols,
    generate_minutes_calendar_from_daily,
)


class BaostockCollector(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="5min",
        max_workers=1,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        bs.login()
        super(BaostockCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_trade_calendar(self):
        _format = "%Y-%m-%d"
        start = self.start_datetime.strftime(_format)
        end = self.end_datetime.strftime(_format)
        # bs.login()
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        calendar_list = []
        while (rs.error_code == '0') & rs.next():
            calendar_list.append(rs.get_row_data())
        calendar_df = pd.DataFrame(calendar_list, columns=rs.fields)
        trade_calendar_df = calendar_df[~calendar_df['is_trading_day'].isin(['0'])]
        # bs.logout()
        return trade_calendar_df['calendar_date'].values

    def normalize_symbol(self, symbol: str):
        pass

    def get_data(self, symbol: str, frequency: str) -> pd.DataFrame:
        data_list = []
        _format = "%Y-%m-%d"
        _start = self.start_datetime.strftime(_format)
        _end = self.end_datetime.strftime(_format)
        rs = bs.query_history_k_data_plus(
            symbol,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=_start,
            end_date=_end,
            frequency=frequency,
            adjustflag="3"
        )
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result["date"] = pd.to_datetime(result["time"].str[:14]) - pd.Timedelta("5min")
        result["symbol"] = result["code"].apply(lambda x: str(x).replace(".", ""))
        result.drop(columns=["time", "code"], inplace=True)
        return result

    def collector_data(self):
        """collector data"""
        # super(BaostockCollector, self).collector_data()
        self.download_data()

    @abc.abstractmethod
    def download_data(self):
        """download data"""
        raise NotImplementedError("rewrite download_data")


class BaostockHS300Collector(BaostockCollector):
    def get_hs300_symbols(self) -> List[str]:
        hs300_stocks = []
        trade_calendar = self.get_trade_calendar()
        with tqdm(total=len(trade_calendar)) as p_bar:
            # bs.login()
            for date in trade_calendar:
                rs = bs.query_hs300_stocks(date=date)
                while rs.error_code == '0' and rs.next():
                    hs300_stocks.append(rs.get_row_data())
                p_bar.update()
            # bs.logout()
        return sorted(set([e[1] for e in hs300_stocks]))

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = self.get_hs300_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols


class BaostockHS300Collector5min(BaostockHS300Collector):
    def download_data(self):
        with tqdm(total=len(self.instrument_list)) as p_bar:
            for symbol in self.instrument_list:
                symbol_data = self.get_data(symbol=symbol, frequency="5")
                save_path = self.save_dir.joinpath(f"{symbol.replace('.', '')}.csv")
                symbol_data.to_csv(save_path, index=False)
                p_bar.update()


class BaostockNormalizeHS3005min(YahooNormalizeCN, YahooNormalize1minOffline):
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # return self.generate_1min_from_daily(self.calendar_list_1d)
        return generate_minutes_calendar_from_daily(
        self.calendar_list_1d, freq="5min", am_range=self.AM_RANGE, pm_range=self.PM_RANGE
        )

    def symbol_to_yahoo(self, symbol):
        if "." not in symbol:
            _exchange = symbol[:2]
            _exchange = ("ss" if _exchange.islower() else "SS") if _exchange.lower() == "sh" else _exchange
            symbol = symbol[2:] + "." + _exchange
        return symbol

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region="HS300"):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        interval: str
            freq, value from [5min], default 1d
        region: str
            region, value from ["HS300"], default "HS300"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"Baostock{self.region.upper()}Collector{self.interval}"

    @property
    def normalize_class_name(self):
        return f"BaostockNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)


if __name__ == "__main__":
    fire.Fire(Run)
