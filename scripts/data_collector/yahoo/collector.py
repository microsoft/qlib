# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
from abc import ABC
from pathlib import Path
from typing import Iterable, Type

import fire
import requests
import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker
from dateutil.tz import tzlocal
from qlib.utils import code_to_fname, fname_to_code
from qlib.config import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import (
    get_calendar_list,
    get_hs_stock_symbols,
    get_us_stock_symbols,
    generate_minutes_calendar_from_daily,
)

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


class YahooCollector(BaseCollector):
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
            stock save dir
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
        super(YahooCollector, self).__init__(
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

        self.init_datetime()

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        # using for 1min
        self._next_datetime = self.convert_datetime(self.start_datetime.date() + pd.Timedelta(days=1), self._timezone)
        self._latest_datetime = self.convert_datetime(self.end_datetime.date(), self._timezone)

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

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end, show_1min_logging: bool = False):
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        def _show_logging_func():
            if interval == YahooCollector.INTERVAL_1min and show_1min_logging:
                logger.warning(f"{error_msg}:{_resp}")

        interval = "1m" if interval in ["1m", "1min"] else interval
        try:
            _resp = Ticker(symbol, asynchronous=False).history(interval=interval, start=start, end=end)
            if isinstance(_resp, pd.DataFrame):
                return _resp.reset_index()
            elif isinstance(_resp, dict):
                _temp_data = _resp.get(symbol, {})
                if isinstance(_temp_data, str) or (
                    isinstance(_resp, dict) and _temp_data.get("indicators", {}).get("quote", None) is None
                ):
                    _show_logging_func()
            else:
                _show_logging_func()
        except Exception as e:
            logger.warning(f"{error_msg}:{e}")

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        def _get_simple(start_, end_):
            self.sleep()
            _remote_interval = "1m" if interval == self.INTERVAL_1min else interval
            return self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )

        _result = None
        if interval == self.INTERVAL_1d:
            _result = _get_simple(start_datetime, end_datetime)
        elif interval == self.INTERVAL_1min:
            if self._next_datetime >= self._latest_datetime:
                _result = _get_simple(start_datetime, end_datetime)
            else:
                _res = []

                def _get_multi(start_, end_):
                    _resp = _get_simple(start_, end_)
                    if _resp is not None and not _resp.empty:
                        _res.append(_resp)

                for _s, _e in (
                    (self.start_datetime, self._next_datetime),
                    (self._latest_datetime, self.end_datetime),
                ):
                    _get_multi(_s, _e)
                for _start in pd.date_range(self._next_datetime, self._latest_datetime, closed="left"):
                    _end = _start + pd.Timedelta(days=1)
                    _get_multi(_start, _end)
                if _res:
                    _result = pd.concat(_res, sort=False).sort_values(["symbol", "date"])
        else:
            raise ValueError(f"cannot support {self.interval}")
        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """collector data"""
        super(YahooCollector, self).collector_data()
        self.download_index_data()

    @abc.abstractmethod
    def download_index_data(self):
        """download index data"""
        raise NotImplementedError("rewrite download_index_data")


class YahooCollectorCN(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class YahooCollectorCN1d(YahooCollectorCN):
    @property
    def min_numbers_trading(self):
        return 252 / 4

    def download_index_data(self):
        # TODO: from MSN
        _format = "%Y%m%d"
        _begin = self.start_datetime.strftime(_format)
        _end = (self.end_datetime + pd.Timedelta(days=-1)).strftime(_format)
        for _index_name, _index_code in {"csi300": "000300", "csi100": "000903"}.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                df = pd.DataFrame(
                    map(
                        lambda x: x.split(","),
                        requests.get(INDEX_BENCH_URL.format(index_code=_index_code, begin=_begin, end=_end)).json()[
                            "data"
                        ]["klines"],
                    )
                )
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            df.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.astype(float, errors="ignore")
            df["adjclose"] = df["close"]
            df["symbol"] = f"sh{_index_code}"
            _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
            if _path.exists():
                _old_df = pd.read_csv(_path)
                df = _old_df.append(df, sort=False)
            df.to_csv(_path, index=False)
            time.sleep(5)


class YahooCollectorCN1min(YahooCollectorCN):
    @property
    def min_numbers_trading(self):
        return 60 * 4 * 5

    def download_index_data(self):
        # TODO: 1m
        logger.warning(f"{self.__class__.__name__} {self.interval} does not support: download_index_data")


class YahooCollectorUS(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get US stock symbols......")
        symbols = get_us_stock_symbols() + [
            "^GSPC",
            "^NDX",
            "^DJI",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "America/New_York"


class YahooCollectorUS1d(YahooCollectorUS):
    @property
    def min_numbers_trading(self):
        return 252 / 4


class YahooCollectorUS1min(YahooCollectorUS):
    @property
    def min_numbers_trading(self):
        return 60 * 6.5 * 5


class YahooNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def normalize_yahoo(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(YahooNormalize.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), set(df.columns) - {symbol_field_name}] = np.nan
        _tmp_series = df["close"].fillna(method="ffill")
        df["change"] = _tmp_series / _tmp_series.shift(1) - 1
        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_yahoo(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # adjusted price
        df = self.adjusted_price(df)
        return df

    @abc.abstractmethod
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """adjusted price"""
        raise NotImplementedError("rewrite adjusted_price")


class YahooNormalize1d(YahooNormalize, ABC):
    DAILY_FORMAT = "%Y-%m-%d"

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjclose" in df:
            df["factor"] = df["adjclose"] / df["close"]
            df["factor"] = df["factor"].fillna(method="ffill")
        else:
            df["factor"] = 1
        for _col in self.COLUMNS:
            if _col not in df.columns:
                continue
            if _col == "volume":
                df[_col] = df[_col] / df["factor"]
            else:
                df[_col] = df[_col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(YahooNormalize1d, self).normalize(df)
        df = self._manual_adj_data(df)
        return df

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """manual adjust data: All fields (except change) are standardized according to the close of the first day"""
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        df = df.loc[df["close"].first_valid_index() :]
        _close = df["close"].iloc[0]
        for _col in df.columns:
            if _col == self._symbol_field_name:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            elif _col != "change":
                df[_col] = df[_col] / _close
            else:
                pass
        return df.reset_index()


class YahooNormalize1min(YahooNormalize, ABC):
    AM_RANGE = None  # type: tuple  # eg: ("09:30:00", "11:29:00")
    PM_RANGE = None  # type: tuple  # eg: ("13:00:00", "14:59:00")

    # Whether the trading day of 1min data is consistent with 1d
    CONSISTENT_1d = False

    def __init__(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
    ):
        """

        Parameters
        ----------
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        super(YahooNormalize1min, self).__init__(date_field_name, symbol_field_name)
        _class_name = self.__class__.__name__.replace("min", "d")
        _class = getattr(importlib.import_module("collector"), _class_name)  # type: Type[YahooNormalize]
        self.data_1d_obj = _class(self._date_field_name, self._symbol_field_name)

    @property
    def calendar_list_1d(self):
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = self._get_1d_calendar_list()
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        return calendar_list_1d

    def generate_1min_from_daily(self, calendars: Iterable) -> pd.Index:
        return generate_minutes_calendar_from_daily(
            calendars, freq="1min", am_range=self.AM_RANGE, pm_range=self.PM_RANGE
        )

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: using daily data factor
        if df.empty:
            return df
        df = df.copy()
        symbol = df.iloc[0][self._symbol_field_name]
        # get 1d data from yahoo
        _start = pd.Timestamp(df[self._date_field_name].min()).strftime(self.DAILY_FORMAT)
        _end = (pd.Timestamp(df[self._date_field_name].max()) + pd.Timedelta(days=1)).strftime(self.DAILY_FORMAT)
        data_1d = YahooCollector.get_data_from_remote(
            self.symbol_to_yahoo(symbol), interval="1d", start=_start, end=_end
        )
        if data_1d is None or data_1d.empty:
            df["factor"] = 1
            # TODO: np.nan or 1 or 0
            df["paused"] = np.nan
        else:
            data_1d = self.data_1d_obj.normalize(data_1d)  # type: pd.DataFrame
            # NOTE: volume is np.nan or volume <= 0, paused = 1
            # FIXME: find a more accurate data source
            data_1d["paused"] = 0
            data_1d.loc[(data_1d["volume"].isna()) | (data_1d["volume"] <= 0), "paused"] = 1
            data_1d = data_1d.set_index(self._date_field_name)

            # add factor from 1d data
            df["date_tmp"] = df[self._date_field_name].apply(lambda x: pd.Timestamp(x).date())
            df.set_index("date_tmp", inplace=True)
            df.loc[:, "factor"] = data_1d["factor"]
            df.loc[:, "paused"] = data_1d["paused"]
            df.reset_index("date_tmp", drop=True, inplace=True)

            if self.CONSISTENT_1d:
                # the date sequence is consistent with 1d
                df.set_index(self._date_field_name, inplace=True)
                df = df.reindex(
                    self.generate_1min_from_daily(
                        pd.to_datetime(data_1d.reset_index()[self._date_field_name].drop_duplicates())
                    )
                )
                df[self._symbol_field_name] = df.loc[df[self._symbol_field_name].first_valid_index()][
                    self._symbol_field_name
                ]
                df.index.names = [self._date_field_name]
                df.reset_index(inplace=True)
        for _col in self.COLUMNS:
            if _col not in df.columns:
                continue
            if _col == "volume":
                df[_col] = df[_col] / df["factor"]
            else:
                df[_col] = df[_col] * df["factor"]
        return df

    @abc.abstractmethod
    def symbol_to_yahoo(self, symbol):
        raise NotImplementedError("rewrite symbol_to_yahoo")

    @abc.abstractmethod
    def _get_1d_calendar_list(self):
        raise NotImplementedError("rewrite _get_1d_calendar_list")


class YahooNormalizeUS:
    def _get_calendar_list(self):
        # TODO: from MSN
        return get_calendar_list("US_ALL")


class YahooNormalizeUS1d(YahooNormalizeUS, YahooNormalize1d):
    pass


class YahooNormalizeUS1min(YahooNormalizeUS, YahooNormalize1min):
    CONSISTENT_1d = False

    def _get_calendar_list(self):
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("US_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class YahooNormalizeCN:
    def _get_calendar_list(self):
        # TODO: from MSN
        return get_calendar_list("ALL")


class YahooNormalizeCN1d(YahooNormalizeCN, YahooNormalize1d):
    pass


class YahooNormalizeCN1min(YahooNormalizeCN, YahooNormalize1min):
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")

    CONSISTENT_1d = True

    def _get_calendar_list(self):
        return self.generate_1min_from_daily(self.calendar_list_1d)

    def symbol_to_yahoo(self, symbol):
        if "." not in symbol:
            _exchange = symbol[:2]
            _exchange = "ss" if _exchange == "sh" else _exchange
            symbol = symbol[2:] + "." + _exchange
        return symbol

    def _get_1d_calendar_list(self):
        return get_calendar_list("ALL")


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=4, interval="1d", region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 4
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN", "US"], default "CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"YahooCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"YahooNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
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
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``
        check_data_length: bool
            check data length, by default False
        limit_nums: int
            using for debug, by default None

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        super(Run, self).download_data(
            max_collector_count, delay, start, end, self.interval, check_data_length, limit_nums
        )

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol"):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        super(Run, self).normalize_data(date_field_name, symbol_field_name)


if __name__ == "__main__":
    fire.Fire(Run)
