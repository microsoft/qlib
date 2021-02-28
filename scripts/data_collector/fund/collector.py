# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
import json
from abc import ABC
from pathlib import Path
from typing import Iterable, Type
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import fire
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from dateutil.tz import tzlocal

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.utils import get_en_fund_symbols

INDEX_BENCH_URL = "http://api.fund.eastmoney.com/f10/lsjz?callback=jQuery_&fundCode={index_code}&pageIndex=1&pageSize={numberOfHistoricalDaysToCrawl}&startDate={startDate}&endDate={endDate}"
REGION_CN = "CN"

class FundData:
    START_DATETIME = pd.Timestamp("2000-01-01")
    END_DATETIME = pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))
    INTERVAL_1d = "1d"

    def __init__(
        self,
        timezone: str = None,
        start=None,
        end=None,
        interval="1d",
        delay=0,
    ):
        """

        Parameters
        ----------
        timezone: str
            The timezone where the data is located
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1d], default 1d
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        """
        self._timezone = tzlocal() if timezone is None else timezone
        self._delay = delay
        self._interval = interval
        self.start_datetime = pd.Timestamp(str(start)) if start else self.START_DATETIME
        self.end_datetime = min(pd.Timestamp(str(end)) if end else self.END_DATETIME, self.END_DATETIME)
        if self._interval != self.INTERVAL_1d:
            raise ValueError(f"interval error: {self._interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt

    def _sleep(self):
        time.sleep(self._delay)

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end):
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        try:
            # TODO: numberOfHistoricalDaysToCrawl should be bigger enouhg
            url = INDEX_BENCH_URL.format(index_code=symbol, numberOfHistoricalDaysToCrawl=10000, startDate=start, endDate=end)
            resp = requests.get(url, headers={"referer": "http://fund.eastmoney.com/110022.html"})

            if resp.status_code != 200:
                raise ValueError("request error")
            
            data = json.loads(resp.text.split("(")[-1].split(")")[0])

            # Some funds don't show the net value, example: http://fundf10.eastmoney.com/jjjz_010288.html
            SYType = data["Data"]["SYType"]
            if (SYType == "每万份收益") or (SYType == "每百份收益") or (SYType == "每百万份收益"):
                raise Exception("The fund contains 每*份收益")

            # TODO: should we sort the value by datetime?
            _resp = pd.DataFrame(data["Data"]["LSJZList"])

            if isinstance(_resp, pd.DataFrame):
                return _resp.reset_index()
        except Exception as e:
            logger.warning(f"{error_msg}:{e}")

    def get_data(self, symbol: str) -> [pd.DataFrame]:
        def _get_simple(start_, end_):
            self._sleep()
            _remote_interval = self._interval
            return self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )

        if self._interval == self.INTERVAL_1d:
            _result = _get_simple(self.start_datetime, self.end_datetime)
        else:
            raise ValueError(f"cannot support {self._interval}")
        return _result


class FundCollector:
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: bool = False,
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            fund save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: bool
            check data length, by default False
        limit_nums: int
            using for debug, by default None
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._delay = delay
        self.max_workers = max_workers
        self._max_collector_count = max_collector_count
        self._mini_symbol_map = {}
        self._interval = interval
        self._check_small_data = check_data_length

        self.fund_list = sorted(set(self.get_fund_list()))
        if limit_nums is not None:
            try:
                self.fund_list = self.fund_list[: int(limit_nums)]
            except Exception as e:
                logger.warning(f"Cannot use limit_nums={limit_nums}, the parameter will be ignored")

        self.fund_data = FundData(
            timezone=self._timezone,
            start=start,
            end=end,
            interval=interval,
            delay=delay,
        )

    @property
    @abc.abstractmethod
    def min_numbers_trading(self):
        # daily, one year: 252 / 4
        # us 1min, a week: 6.5 * 60 * 5
        # cn 1min, a week: 4 * 60 * 5
        raise NotImplementedError("rewrite min_numbers_trading")

    @abc.abstractmethod
    def get_fund_list(self):
        raise NotImplementedError("rewrite get_fund_list")

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    def save_fund(self, symbol, df: pd.DataFrame):
        """save fund data to file

        Parameters
        ----------
        symbol: str
            fund code
        df : pd.DataFrame
            df.columns must contain "symbol" and "datetime"
        """
        if df.empty:
            logger.warning(f"{symbol} is empty")
            return

        fund_path = self.save_dir.joinpath(f"{symbol}.csv")
        df["symbol"] = symbol
        if fund_path.exists():
            # TODO: read the fund code as str, not int, like "000001" shouldn't be "1"
            _old_df = pd.read_csv(fund_path)
            # TODO: remove the duplicate date
            df = _old_df.append(df, sort=False)
        df.to_csv(fund_path, index=False)

    def _save_small_data(self, symbol, df):
        if len(df) <= self.min_numbers_trading:
            logger.warning(f"the number of trading days of {symbol} is less than {self.min_numbers_trading}!")
            _temp = self._mini_symbol_map.setdefault(symbol, [])
            _temp.append(df.copy())
            return None
        else:
            if symbol in self._mini_symbol_map:
                self._mini_symbol_map.pop(symbol)
            return symbol

    def _get_data(self, symbol):
        _result = None
        df = self.fund_data.get_data(symbol)
        if isinstance(df, pd.DataFrame):
            if not df.empty:
                if self._check_small_data:
                    if self._save_small_data(symbol, df) is not None:
                        _result = symbol
                        self.save_fund(symbol, df)
                else:
                    _result = symbol
                    self.save_fund(symbol, df)
        return _result

    def _collector(self, fund_list):
        error_symbol = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=len(fund_list)) as p_bar:
                for _symbol, _result in zip(fund_list, executor.map(self._get_data, fund_list)):
                    if _result is None:
                        error_symbol.append(_symbol)
                    p_bar.update()
        print(error_symbol)
        logger.info(f"error symbol nums: {len(error_symbol)}")
        logger.info(f"current get symbol nums: {len(fund_list)}")
        error_symbol.extend(self._mini_symbol_map.keys())
        return sorted(set(error_symbol))

    def collector_data(self):
        """collector data"""
        logger.info("start collector fund data......")
        fund_list = self.fund_list
        for i in range(self._max_collector_count):
            if not fund_list:
                break
            logger.info(f"getting data: {i+1}")
            fund_list = self._collector(fund_list)
            logger.info(f"{i+1} finish.")
        for _symbol, _df_list in self._mini_symbol_map.items():
            self.save_fund(_symbol, pd.concat(_df_list, sort=False).drop_duplicates(["date"]).sort_values(["date"]))
        if self._mini_symbol_map:
            logger.warning(f"less than {self.min_numbers_trading} fund list: {list(self._mini_symbol_map.keys())}")
        logger.info(f"total {len(self.fund_list)}, error: {len(set(fund_list))}")

class FundollectorCN(FundCollector, ABC):
    def get_fund_list(self):
        logger.info("get cn fund symbols......")
        symbols = get_en_fund_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class FundCollectorCN1d(FundollectorCN):
    @property
    def min_numbers_trading(self):
        return 252 / 4

class Run:
    def __init__(self, source_dir=None, max_workers=4, region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        max_workers: int
            Concurrent number, default is 4
        region: str
            region, value from ["CN"], default "CN"
        """
        if source_dir is None:
            source_dir = CUR_DIR.joinpath("source")
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.source_dir.mkdir(parents=True, exist_ok=True)

        self._cur_module = importlib.import_module("collector")
        self.max_workers = max_workers
        self.region = region

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        interval="1d",
        check_data_length=False,
        limit_nums=None,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1d
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``
        check_data_length: bool # if this param useful?
            check data length, by default False
        limit_nums: int
            using for debug, by default None

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/fund_data/source/cn_1d --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
        """

        _class = getattr(
            self._cur_module, f"FundCollector{self.region.upper()}{interval}"
        )  # type: Type[FundCollector]
        _class(
            self.source_dir,
            max_workers=self.max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        ).collector_data()

if __name__ == "__main__":
    fire.Fire(Run)
