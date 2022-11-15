# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from re import I
import sys
import copy
import time
import datetime
import importlib
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
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
    get_us_stock_symbols,
    get_in_stock_symbols,
    get_br_stock_symbols,
    generate_minutes_calendar_from_daily,
)

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


class YahooCollector(BaseCollector):
    retry = 5  # Configuration attribute.  How many times will it try to re-request the data if the network fails.

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
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
        check_data_length: int
            check data length, by default None
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
            logger.warning(
                f"get data error: {symbol}--{start}--{end}"
                + "Your data request fails. This may be caused by your firewall (e.g. GFW). Please switch your network if you want to access Yahoo! data"
            )

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        @deco_retry(retry_sleep=self.delay, retry=self.retry)
        def _get_simple(start_, end_):
            self.sleep()
            _remote_interval = "1m" if interval == self.INTERVAL_1min else interval
            resp = self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )
            if resp is None or resp.empty:
                raise ValueError(
                    f"get data error: {symbol}--{start_}--{end_}" + "The stock may be delisted, please check"
                )
            return resp

        _result = None
        if interval == self.INTERVAL_1d:
            try:
                _result = _get_simple(start_datetime, end_datetime)
            except ValueError as e:
                pass
        elif interval == self.INTERVAL_1min:
            _res = []
            _start = self.start_datetime
            while _start < self.end_datetime:
                _tmp_end = min(_start + pd.Timedelta(days=7), self.end_datetime)
                try:
                    _resp = _get_simple(_start, _tmp_end)
                    _res.append(_resp)
                except ValueError as e:
                    pass
                _start = _tmp_end
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
                df = pd.concat([_old_df, df], sort=False)
            df.to_csv(_path, index=False)
            time.sleep(5)


class YahooCollectorCN1min(YahooCollectorCN):
    def get_instrument_list(self):
        symbols = super(YahooCollectorCN1min, self).get_instrument_list()
        return symbols + ["000300.ss", "000905.ss", "000903.ss"]

    def download_index_data(self):
        pass


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
    pass


class YahooCollectorUS1min(YahooCollectorUS):
    pass


class YahooCollectorIN(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get INDIA stock symbols......")
        symbols = get_in_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "Asia/Kolkata"


class YahooCollectorIN1d(YahooCollectorIN):
    pass


class YahooCollectorIN1min(YahooCollectorIN):
    pass


class YahooCollectorBR(YahooCollector, ABC):
    def retry(cls):
        """
        The reason to use retry=2 is due to the fact that
        Yahoo Finance unfortunately does not keep track of some
        Brazilian stocks.

        Therefore, the decorator deco_retry with retry argument
        set to 5 will keep trying to get the stock data up to 5 times,
        which makes the code to download Brazilians stocks very slow.

        In future, this may change, but for now
        I suggest to leave retry argument to 1 or 2 in
        order to improve download speed.

        To achieve this goal an abstract attribute (retry)
        was added into YahooCollectorBR base class
        """
        raise NotImplementedError

    def get_instrument_list(self):
        logger.info("get BR stock symbols......")
        symbols = get_br_stock_symbols() + [
            "^BVSP",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "Brazil/East"


class YahooCollectorBR1d(YahooCollectorBR):
    retry = 2
    pass


class YahooCollectorBR1min(YahooCollectorBR):
    retry = 2
    pass


class YahooNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].fillna(method="ffill")
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_yahoo(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
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
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan

        change_series = YahooNormalize.calc_change(df, last_close)
        # NOTE: The data obtained by Yahoo finance sometimes has exceptions
        # WARNING: If it is normal for a `symbol(exchange)` to differ by a factor of *89* to *111* for consecutive trading days,
        # WARNING: the logic in the following line needs to be modified
        _count = 0
        while True:
            # NOTE: may appear unusual for many days in a row
            change_series = YahooNormalize.calc_change(df, last_close)
            _mask = (change_series >= 89) & (change_series <= 111)
            if not _mask.any():
                break
            _tmp_cols = ["high", "close", "low", "open", "adjclose"]
            df.loc[_mask, _tmp_cols] = df.loc[_mask, _tmp_cols] / 100
            _count += 1
            if _count >= 10:
                _symbol = df.loc[df[symbol_field_name].first_valid_index()]["symbol"]
                logger.warning(
                    f"{_symbol} `change` is abnormal for {_count} consecutive days, please check the specific data file carefully"
                )

        df["change"] = YahooNormalize.calc_change(df, last_close)

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

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """get first close value

        Notes
        -----
            For incremental updates(append) to Yahoo 1D data, user need to use a close that is not 0 on the first trading day of the existing data
        """
        df = df.loc[df["close"].first_valid_index() :]
        _close = df["close"].iloc[0]
        return _close

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """manual adjust data: All fields (except change) are standardized according to the close of the first day"""
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        _close = self._get_first_close(df)
        for _col in df.columns:
            # NOTE: retain original adjclose, required for incremental updates
            if _col in [self._symbol_field_name, "adjclose", "change"]:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close
        return df.reset_index()


class YahooNormalize1dExtend(YahooNormalize1d):
    def __init__(
        self, old_qlib_data_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        old_qlib_data_dir: str, Path
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        super(YahooNormalize1dExtend, self).__init__(date_field_name, symbol_field_name)
        self._first_close_field = "first_close"
        self._ori_close_field = "ori_close"
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        import qlib
        from qlib.data import D

        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$close/$factor", "$adjclose/$close"])
        df.columns = [self._ori_close_field, self._first_close_field]
        return df

    def _get_close(self, df: pd.DataFrame, field_name: str):
        _symbol = df.loc[df[self._symbol_field_name].first_valid_index()][self._symbol_field_name].upper()
        _df = self.old_qlib_data.loc(axis=0)[_symbol]
        _close = _df.loc[_df.last_valid_index()][field_name]
        return _close

    def _get_first_close(self, df: pd.DataFrame) -> float:
        try:
            _close = self._get_close(df, field_name=self._first_close_field)
        except KeyError:
            _close = super(YahooNormalize1dExtend, self)._get_first_close(df)
        return _close

    def _get_last_close(self, df: pd.DataFrame) -> float:
        try:
            _close = self._get_close(df, field_name=self._ori_close_field)
        except KeyError:
            _close = None
        return _close

    def _get_last_date(self, df: pd.DataFrame) -> pd.Timestamp:
        _symbol = df.loc[df[self._symbol_field_name].first_valid_index()][self._symbol_field_name].upper()
        try:
            _df = self.old_qlib_data.loc(axis=0)[_symbol]
            _date = _df.index.max()
        except KeyError:
            _date = None
        return _date

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        _last_close = self._get_last_close(df)
        # reindex
        _last_date = self._get_last_date(df)
        if _last_date is not None:
            df = df.set_index(self._date_field_name)
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="first")]
            _max_date = df.index.max()
            df = df.reindex(self._calendar_list).loc[:_max_date].reset_index()
            df = df[df[self._date_field_name] > _last_date]
            if df.empty:
                return pd.DataFrame()
            _si = df["close"].first_valid_index()
            if _si > df.index[0]:
                logger.warning(
                    f"{df.loc[_si][self._symbol_field_name]} missing data: {df.loc[:_si - 1][self._date_field_name].to_list()}"
                )
        # normalize
        df = self.normalize_yahoo(
            df, self._calendar_list, self._date_field_name, self._symbol_field_name, last_close=_last_close
        )
        # adjusted price
        df = self.adjusted_price(df)
        df = self._manual_adj_data(df)
        return df


class YahooNormalize1min(YahooNormalize, ABC):
    AM_RANGE = None  # type: tuple  # eg: ("09:30:00", "11:29:00")
    PM_RANGE = None  # type: tuple  # eg: ("13:00:00", "14:59:00")

    # Whether the trading day of 1min data is consistent with 1d
    CONSISTENT_1d = True
    CALC_PAUSED_NUM = True

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

    def get_1d_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """get 1d data

        Returns
        ------
            data_1d: pd.DataFrame
                data_1d.columns = [self._date_field_name, self._symbol_field_name, "paused", "volume", "factor", "close"]

        """
        data_1d = YahooCollector.get_data_from_remote(self.symbol_to_yahoo(symbol), interval="1d", start=start, end=end)
        if not (data_1d is None or data_1d.empty):
            _class_name = self.__class__.__name__.replace("min", "d")
            _class: type(YahooNormalize) = getattr(importlib.import_module("collector"), _class_name)
            data_1d_obj = _class(self._date_field_name, self._symbol_field_name)
            data_1d = data_1d_obj.normalize(data_1d)
        return data_1d

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: using daily data factor
        if df.empty:
            return df
        df = df.copy()
        df = df.sort_values(self._date_field_name)
        symbol = df.iloc[0][self._symbol_field_name]
        # get 1d data from yahoo
        _start = pd.Timestamp(df[self._date_field_name].min()).strftime(self.DAILY_FORMAT)
        _end = (pd.Timestamp(df[self._date_field_name].max()) + pd.Timedelta(days=1)).strftime(self.DAILY_FORMAT)
        data_1d: pd.DataFrame = self.get_1d_data(symbol, _start, _end)
        data_1d = data_1d.copy()
        if data_1d is None or data_1d.empty:
            df["factor"] = 1 / df.loc[df["close"].first_valid_index()]["close"]
            # TODO: np.nan or 1 or 0
            df["paused"] = np.nan
        else:
            # NOTE: volume is np.nan or volume <= 0, paused = 1
            # FIXME: find a more accurate data source
            data_1d["paused"] = 0
            data_1d.loc[(data_1d["volume"].isna()) | (data_1d["volume"] <= 0), "paused"] = 1
            data_1d = data_1d.set_index(self._date_field_name)

            # add factor from 1d data
            # NOTE: yahoo 1d data info:
            #   - Close price adjusted for splits. Adjusted close price adjusted for both dividends and splits.
            #   - data_1d.adjclose: Adjusted close price adjusted for both dividends and splits.
            #   - data_1d.close: `data_1d.adjclose / (close for the first trading day that is not np.nan)`
            def _calc_factor(df_1d: pd.DataFrame):
                try:
                    _date = pd.Timestamp(pd.Timestamp(df_1d[self._date_field_name].iloc[0]).date())
                    df_1d["factor"] = (
                        data_1d.loc[_date]["close"] / df_1d.loc[df_1d["close"].last_valid_index()]["close"]
                    )
                    df_1d["paused"] = data_1d.loc[_date]["paused"]
                except Exception:
                    df_1d["factor"] = np.nan
                    df_1d["paused"] = np.nan
                return df_1d

            df = df.groupby([df[self._date_field_name].dt.date]).apply(_calc_factor)

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

        if self.CALC_PAUSED_NUM:
            df = self.calc_paused_num(df)
        return df

    def calc_paused_num(self, df: pd.DataFrame):
        _symbol = df.iloc[0][self._symbol_field_name]
        df = df.copy()
        df["_tmp_date"] = df[self._date_field_name].apply(lambda x: pd.Timestamp(x).date())
        # remove data that starts and ends with `np.nan` all day
        all_data = []
        # Record the number of consecutive trading days where the whole day is nan, to remove the last trading day where the whole day is nan
        all_nan_nums = 0
        # Record the number of consecutive occurrences of trading days that are not nan throughout the day
        not_nan_nums = 0
        for _date, _df in df.groupby("_tmp_date"):
            _df["paused"] = 0
            if not _df.loc[_df["volume"] < 0].empty:
                logger.warning(f"volume < 0, will fill np.nan: {_date} {_symbol}")
                _df.loc[_df["volume"] < 0, "volume"] = np.nan

            check_fields = set(_df.columns) - {
                "_tmp_date",
                "paused",
                "factor",
                self._date_field_name,
                self._symbol_field_name,
            }
            if _df.loc[:, check_fields].isna().values.all() or (_df["volume"] == 0).all():
                all_nan_nums += 1
                not_nan_nums = 0
                _df["paused"] = 1
                if all_data:
                    _df["paused_num"] = not_nan_nums
                    all_data.append(_df)
            else:
                all_nan_nums = 0
                not_nan_nums += 1
                _df["paused_num"] = not_nan_nums
                all_data.append(_df)
        all_data = all_data[: len(all_data) - all_nan_nums]
        if all_data:
            df = pd.concat(all_data, sort=False)
        else:
            logger.warning(f"data is empty: {_symbol}")
            df = pd.DataFrame()
            return df
        del df["_tmp_date"]
        return df

    @abc.abstractmethod
    def symbol_to_yahoo(self, symbol):
        raise NotImplementedError("rewrite symbol_to_yahoo")

    @abc.abstractmethod
    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        raise NotImplementedError("rewrite _get_1d_calendar_list")


class YahooNormalize1minOffline(YahooNormalize1min):
    """Normalised to 1min using local 1d data"""

    def __init__(
        self, qlib_data_1d_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        qlib_data_1d_dir: str, Path
            the qlib data to be updated for yahoo, usually from: Normalised to 1min using local 1d data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        self.qlib_data_1d_dir = qlib_data_1d_dir
        super(YahooNormalize1minOffline, self).__init__(date_field_name, symbol_field_name)
        self._all_1d_data = self._get_all_1d_data()

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        import qlib
        from qlib.data import D

        qlib.init(provider_uri=self.qlib_data_1d_dir)
        return list(D.calendar(freq="day"))

    def _get_all_1d_data(self):
        import qlib
        from qlib.data import D

        qlib.init(provider_uri=self.qlib_data_1d_dir)
        df = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")
        df.reset_index(inplace=True)
        df.rename(columns={"datetime": self._date_field_name, "instrument": self._symbol_field_name}, inplace=True)
        df.columns = list(map(lambda x: x[1:] if x.startswith("$") else x, df.columns))
        return df

    def get_1d_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """get 1d data

        Returns
        ------
            data_1d: pd.DataFrame
                data_1d.columns = [self._date_field_name, self._symbol_field_name, "paused", "volume", "factor", "close"]

        """
        return self._all_1d_data[
            (self._all_1d_data[self._symbol_field_name] == symbol.upper())
            & (self._all_1d_data[self._date_field_name] >= pd.Timestamp(start))
            & (self._all_1d_data[self._date_field_name] < pd.Timestamp(end))
        ]


class YahooNormalizeUS:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: from MSN
        return get_calendar_list("US_ALL")


class YahooNormalizeUS1d(YahooNormalizeUS, YahooNormalize1d):
    pass


class YahooNormalizeUS1dExtend(YahooNormalizeUS, YahooNormalize1dExtend):
    pass


class YahooNormalizeUS1min(YahooNormalizeUS, YahooNormalize1minOffline):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("US_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class YahooNormalizeIN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("IN_ALL")


class YahooNormalizeIN1d(YahooNormalizeIN, YahooNormalize1d):
    pass


class YahooNormalizeIN1min(YahooNormalizeIN, YahooNormalize1minOffline):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("IN_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class YahooNormalizeCN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: from MSN
        return get_calendar_list("ALL")


class YahooNormalizeCN1d(YahooNormalizeCN, YahooNormalize1d):
    pass


class YahooNormalizeCN1dExtend(YahooNormalizeCN, YahooNormalize1dExtend):
    pass


class YahooNormalizeCN1min(YahooNormalizeCN, YahooNormalize1minOffline):
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return self.generate_1min_from_daily(self.calendar_list_1d)

    def symbol_to_yahoo(self, symbol):
        if "." not in symbol:
            _exchange = symbol[:2]
            _exchange = ("ss" if _exchange.islower() else "SS") if _exchange.lower() == "sh" else _exchange
            symbol = symbol[2:] + "." + _exchange
        return symbol

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")


class YahooNormalizeBR:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("BR_ALL")


class YahooNormalizeBR1d(YahooNormalizeBR, YahooNormalize1d):
    pass


class YahooNormalizeBR1min(YahooNormalizeBR, YahooNormalize1minOffline):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("BR_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region=REGION_CN):
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
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN", "US", "BR"], default "CN"
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
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        start: str
            start datetime, default "2000-01-01"; closed interval(including start)
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        end_date: str
            if not None, normalize the last date saved (including end_date); if None, it will ignore this parameter; by default None
        qlib_data_1d_dir: str
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1d data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                or:
                    download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region cn --interval 1d
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source_cn_1min --normalize_dir ~/.qlib/stock_data/normalize_cn_1min --region CN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
                )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )

    def normalize_data_1d_extend(
        self, old_qlib_data_dir, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """normalize data extend; extending yahoo qlib data(from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data)

        Notes
        -----
            Steps to extend yahoo qlib data:

                1. download qlib data: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data; save to <dir1>

                2. collector source data: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#collector-data; save to <dir2>

                3. normalize new source data(from step 2): python scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1d

                4. dump data: python scripts/dump_bin.py dump_update --csv_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date

                5. update instrument(eg. csi300): python python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir <dir1> --method parse_instruments

        Parameters
        ----------
        old_qlib_data_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
        )
        yc.normalize()

    def download_today_data(
        self,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
    ):
        """download today data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            Download today's data:
                start_time = datetime.datetime.now().date(); closed interval(including start)
                end_time = pd.Timestamp(start_time + pd.Timedelta(days=1)).date(); open interval(excluding end)

            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1m
        """
        start = datetime.datetime.now().date()
        end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
        self.download_data(
            max_collector_count,
            delay,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            check_data_length,
            limit_nums,
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        trading_date: str = None,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 1,
    ):
        """update yahoo data to bin

        Parameters
        ----------
        qlib_data_1d_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data

        trading_date: str
            trading days to be updated, by default ``datetime.datetime.now().strftime("%Y-%m-%d")``
        end_date: str
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        delay: float
            time.sleep(delay), default 1
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
            # get 1m data
        """

        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")

        # start/end date
        if trading_date is None:
            trading_date = datetime.datetime.now().strftime("%Y-%m-%d")
            logger.warning(f"trading_date is None, use the current date: {trading_date}")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # download qlib 1d data
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region)

        # download data from yahoo
        # NOTE: when downloading data from YahooFinance, max_workers is recommended to be 1
        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)
        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        # normalize data
        self.normalize_data_1d_extend(qlib_data_1d_dir)

        # dump bin
        _dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # parse index
        _region = self.region.lower()
        if _region not in ["cn", "us"]:
            logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
            return
        index_list = ["CSI100", "CSI300"] if _region == "cn" else ["SP500", "NASDAQ100", "DJIA", "SP400"]
        get_instruments = getattr(
            importlib.import_module(f"data_collector.{_region}_index.collector"), "get_instruments"
        )
        for _index in index_list:
            get_instruments(str(qlib_data_1d_dir), _index, market_index=f"{_region}_index")


if __name__ == "__main__":
    fire.Fire(Run)
