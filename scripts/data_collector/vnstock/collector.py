# Copyright (c) ManhTran76

import abc
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
from vnstock import Vnstock, Listing, Quote
from dateutil.tz import tzlocal

import qlib
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_VN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    generate_minutes_calendar_from_daily,
    calc_adjusted_price,
)

def get_vn_stock_symbols():
    """Get Vietnamese stock symbols using vnstock"""
    try:
        listing = Listing()
        symbols_df = listing.all_symbols()
        if isinstance(symbols_df, pd.DataFrame) and not symbols_df.empty:
            # Assuming the symbol column is named 'symbol' or 'ticker'
            symbol_col = None
            for col in ['symbol', 'ticker', 'code']:
                if col in symbols_df.columns:
                    symbol_col = col
                    break
            
            if symbol_col is not None:
                symbols = symbols_df[symbol_col].tolist()
                return symbols
            else:
                logger.warning("Cannot find symbol column in the data")
                return []
        else:
            logger.warning("No symbol data available")
            return []
    except Exception as e:
        logger.warning(f"Error getting symbols: {e}")
        return []


class VNStockCollector(BaseCollector, ABC):
    retry = 2  # Configuration attribute.  How many times will it try to re-request the data if the network fails.

    def __init__(
        self,
        save_dir: str | Path,
        start=None,
        end=None,
        interval="1D",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int | None = None,
        limit_nums: int | None = None,
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
            freq, value from [1min, 1D], default 1min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        super(VNStockCollector, self).__init__(
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
        if self.interval == self.INTERVAL_1min or self.interval == self.INTERVAL_1m:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d or self.interval == self.INTERVAL_1D:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: pd.Timestamp | datetime.date | str, timezone):
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
            if interval == VNStockCollector.INTERVAL_1min and show_1min_logging:
                logger.warning(f"{error_msg}: Error fetching data")

        try:
            # Convert interval to vnstock format
            vnstock_interval = "1m" if interval in ["1m", "1min"] else interval
            
            # Create vnstock Quote object
            quote = Quote(symbol=symbol, source='TCBS')
            # convert datetime to str if needed
            if isinstance(start, pd.Timestamp):
                start = start.strftime("%Y-%m-%d")
            if isinstance(end, pd.Timestamp):
                end = end.strftime("%Y-%m-%d")
            _resp = quote.history(start=start, end=end, interval=vnstock_interval)
            
            # if there is 'date' column, change to 'time'
            if 'date' in _resp.columns:
                _resp = _resp.rename(columns={'date': 'time'})

            if isinstance(_resp, pd.DataFrame) and not _resp.empty:
                # Add symbol column if not present
                if 'symbol' not in _resp.columns:
                    _resp['symbol'] = symbol
                return _resp.reset_index()
            else:
                _show_logging_func()
                return None
        except Exception as e:
            logger.warning(
                f"get data error: {symbol}--{start}--{end}--{e}. "
                + "This may be caused by network issues or invalid symbol. Please check the symbol and try again."
            )
            return None

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
        if interval == self.INTERVAL_1d or interval == self.INTERVAL_1D:
            try:
                _result = _get_simple(start_datetime, end_datetime)
            except ValueError as e:
                pass
        elif interval == self.INTERVAL_1min or interval == self.INTERVAL_1m:
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
        super(VNStockCollector, self).collector_data()
        self.download_index_data()

    @abc.abstractmethod
    def download_index_data(self):
        """download index data"""
        raise NotImplementedError("rewrite download_index_data")

    def get_instrument_list(self):
        logger.info("get VN stock symbols......")
        symbols = get_vn_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        # Vietnamese stock symbols are typically just the symbol itself
        # No need for prefix like Chinese stocks (sh/sz)
        return symbol.upper()

    @property
    def _timezone(self):
        return "Asia/Ho_Chi_Minh"


class VNStockCollectorVN1D(VNStockCollector):
    def download_index_data(self):
        # Download Vietnamese index data (VNINDEX, HNXINDEX, UPCOMINDEX)
        logger.info("Downloading Vietnamese index data...")
        for _index_name, _index_code in {"vnindex": "VNINDEX", "hnxindex": "HNXINDEX", "upcomindex": "UPCOMINDEX"}.items():
            logger.info(f"get index data: {_index_name}({_index_code})......")
            try:
                quote = Quote(symbol=_index_code, source='TCBS')
                # convert datetime to str if needed
                if isinstance(self.start_datetime, pd.Timestamp):
                    self.start_datetime = self.start_datetime.strftime("%Y-%m-%d")
                if isinstance(self.end_datetime, pd.Timestamp):
                    self.end_datetime = self.end_datetime.strftime("%Y-%m-%d")
                df = quote.history(start=self.start_datetime, end=self.end_datetime, interval='1D')
                
                if df is not None and not df.empty:
                    # Rename columns to match qlib format
                    column_mapping = {
                        'time': 'date',
                        'open': 'open', 
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume'
                    }
                    
                    # Apply column mapping if columns exist
                    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                    
                    # Ensure required columns exist
                    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                    for col in required_cols:
                        if col not in df.columns:
                            if col == 'volume':
                                df[col] = 0  # Default volume for indices
                            else:
                                logger.warning(f"Missing column {col} for index {_index_code}")
                    
                    # Add adjclose and symbol columns
                    df["adjclose"] = df["close"]
                    df["symbol"] = _index_code.lower()
                    
                    # Save to file
                    _path = self.save_dir.joinpath(f"{_index_code.lower()}.csv")
                    if _path.exists():
                        _old_df = pd.read_csv(_path)
                        df = pd.concat([_old_df, df], sort=False)
                        df = df.drop_duplicates(subset=['date'], keep='last')
                    
                    df.to_csv(_path, index=False)
                    logger.info(f"Saved {len(df)} records for {_index_code}")
                else:
                    logger.warning(f"No data received for {_index_code}")
                    
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            time.sleep(1)  # Add delay between requests


class VNStockCollectorVN1min(VNStockCollector):
    def get_instrument_list(self):
        symbols = super(VNStockCollectorVN1min, self).get_instrument_list()
        # Add Vietnamese indices for 1min data
        return symbols + ["VNINDEX", "HNXINDEX", "UPCOMINDEX"]

    def download_index_data(self):
        pass


class VNStockNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float | None) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].ffill()
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_yahoo(
        df: pd.DataFrame,
        calendar_list: list | None = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float | None = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(VNStockNormalize.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index, format='mixed')
        df.index = df.index.tz_localize(None)
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

        change_series = VNStockNormalize.calc_change(df, last_close)
        # NOTE: The data obtained by VNStock finance sometimes has exceptions
        # WARNING: If it is normal for a `symbol(exchange)` to differ by a factor of *89* to *111* for consecutive trading days,
        # WARNING: the logic in the following line needs to be modified
        _count = 0
        while True:
            # NOTE: may appear unusual for many days in a row
            change_series = VNStockNormalize.calc_change(df, last_close)
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

        df["change"] = VNStockNormalize.calc_change(df, last_close)

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


class VNStockNormalize1D(VNStockNormalize, ABC):
    DAILY_FORMAT = "%Y-%m-%d"

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjclose" in df:
            df["factor"] = df["adjclose"] / df["close"]
            df["factor"] = df["factor"].ffill()
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
        df = super(VNStockNormalize1D, self).normalize(df)
        df = self._manual_adj_data(df)
        return df

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """get first close value

        Notes
        -----
            For incremental updates(append) to VNStock 1D data, user need to use a close that is not 0 on the first trading day of the existing data
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


class VNStockNormalize1DExtend(VNStockNormalize1D):
    def __init__(
        self, old_qlib_data_dir: str | Path, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
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
        super(VNStockNormalize1DExtend, self).__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: str | Path):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(VNStockNormalize1DExtend, self).normalize(df)
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        df = df.loc[latest_date:]
        new_latest_data = df.iloc[0]
        old_latest_data = old_df.loc[latest_date]
        for col in self.column_list[:-1]:
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        return df.drop(df.index[0]).reset_index()


class VNStockNormalize1min(VNStockNormalize, ABC):
    """Normalised to 1min using local 1D data"""

    AM_RANGE: tuple | None = None  # eg: ("09:30:00", "11:29:00")
    PM_RANGE: tuple | None = None  # eg: ("13:00:00", "14:59:00")

    # Whether the trading day of 1min data is consistent with 1D
    CONSISTENT_1d = True
    CALC_PAUSED_NUM = True

    def __init__(
        self, qlib_data_1d_dir: str | Path, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        qlib_data_1d_dir: str, Path
            the qlib data to be updated for yahoo, usually from: Normalised to 1min using local 1D data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        super(VNStockNormalize1min, self).__init__(date_field_name, symbol_field_name)
        qlib.init(provider_uri=qlib_data_1d_dir)
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return list(D.calendar(freq="day"))

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
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="1min",
            consistent_1d=self.CONSISTENT_1d,
            calc_paused=self.CALC_PAUSED_NUM,
            _1d_data_all=self.all_1d_data,
        )
        return df

    @abc.abstractmethod
    def symbol_to_yahoo(self, symbol):
        raise NotImplementedError("rewrite symbol_to_yahoo")

    @abc.abstractmethod
    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        raise NotImplementedError("rewrite _get_1d_calendar_list")

class VNStockNormalizeVN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """Get Vietnamese trading calendar from vnstock
        
        Returns
        -------
        Iterable[pd.Timestamp]
            Vietnamese trading calendar dates
        """
        try:
            # Use VNINDEX data to generate trading calendar
            vnstock = Vnstock()
            stock = vnstock.stock(symbol="VNI", source="TCBS")  # VNINDEX
            
            # Get historical data to determine trading days
            start_date = "2000-01-01"
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            
            df = stock.quote.history(start=start_date, end=end_date, interval="1D")
            
            if df is not None and not df.empty:
                # Extract trading dates from the time column
                trading_dates = pd.to_datetime(df['time'], format='mixed').dt.date
                calendar = sorted([pd.Timestamp(date) for date in trading_dates.unique()])
                logger.info(f"Successfully retrieved {len(calendar)} Vietnamese trading dates from vnstock")
                return calendar
            else:
                logger.warning("No data returned from vnstock for Vietnamese calendar")
                return self._get_fallback_calendar()
                
        except Exception as e:
            logger.warning(f"Error getting Vietnamese calendar from vnstock: {e}")
            return self._get_fallback_calendar()
    
    def _get_fallback_calendar(self) -> Iterable[pd.Timestamp]:
        """Generate fallback Vietnamese trading calendar based on business days
        
        Returns
        -------
        Iterable[pd.Timestamp]
            Fallback trading calendar
        """
        logger.info("Using fallback Vietnamese trading calendar based on business days")
        
        # Generate business days (Monday to Friday)
        calendar = pd.bdate_range(start="2000-01-01", end=pd.Timestamp.now(), freq='B')
        
        # Convert to list of Timestamps
        calendar_list = [pd.Timestamp(date) for date in calendar]
        
        # Filter out known Vietnamese holidays (simplified)
        # Note: This is a basic implementation. For production use, 
        # consider using a proper Vietnamese holiday calendar library
        filtered_calendar = []
        for date in calendar_list:
            # Skip some major Vietnamese holidays (simplified check)
            if self._is_vietnamese_holiday(date):
                continue
            filtered_calendar.append(date)
        
        return filtered_calendar
    
    def _is_vietnamese_holiday(self, date: pd.Timestamp) -> bool:
        """Check if a date is a Vietnamese holiday (simplified implementation)
        
        Parameters
        ----------
        date : pd.Timestamp
            Date to check
            
        Returns
        -------
        bool
            True if the date is a Vietnamese holiday
        """
        # Simplified holiday check - only checking fixed holidays
        month = date.month
        day = date.day
        
        # New Year's Day
        if month == 1 and day == 1:
            return True
        
        # International Labor Day
        if month == 5 and day == 1:
            return True
            
        # Liberation Day
        if month == 4 and day == 30:
            return True
            
        # National Day
        if month == 9 and day == 2:
            return True
        
        # Note: Tet holidays and other lunar calendar holidays would need 
        # more sophisticated calculation, as they vary each year
        
        return False


class VNStockNormalizeVN1D(VNStockNormalizeVN, VNStockNormalize1D):
    pass


class VNStockNormalizeVN1DExtend(VNStockNormalizeVN, VNStockNormalize1DExtend):
    pass


class VNStockNormalizeVN1min(VNStockNormalizeVN, VNStockNormalize1min):
    AM_RANGE = ("09:00:00", "11:30:00")  # Vietnamese market hours
    PM_RANGE = ("13:00:00", "15:00:00")  # Vietnamese market hours

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return self.generate_1min_from_daily(self.calendar_list_1d)

    def symbol_to_yahoo(self, symbol):
        # Vietnamese stocks don't need conversion like Chinese stocks
        # VNStock uses direct symbol names
        return symbol

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("VN_ALL")

class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1D", region="VN"):
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
            freq, value from [1min, 1D], default 1D
        region: str
            region, value from ["VN"], default "VN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"VNStockCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"VNStockNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> Path | str:
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
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region VN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1D
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region VN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        if self.interval == "1D" and end is not None and pd.Timestamp(end) > pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d")):
            raise ValueError(f"end_date: {end} is greater than the current date.")

        super(Run, self).download_data(max_collector_count, int(delay), start, end, check_data_length or 0, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str | None = None,
        qlib_data_1d_dir: str | None = None,
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
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1D data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1D
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                or:
                    download 1D data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1D-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region vn --interval 1D
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/vn_data --source_dir ~/.qlib/stock_data/source_vn_1min --normalize_dir ~/.qlib/stock_data/normalize_vn_1min --region VN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1D data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
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

                3. normalize new source data(from step 2): python scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1D

                4. dump data: python scripts/dump_bin.py dump_update --data_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date

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
            $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/vn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region VN --interval 1D
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
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region VN --delay 0.1 --interval 1D
            # get 1m data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region VN --delay 0.1 --interval 1m
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
        end_date: str | None = None,
        check_data_length: int | None = None,
        delay: float = 1,
        exists_skip: bool = False,
    ):
        """update yahoo data to bin

        Parameters
        ----------
        qlib_data_1d_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data

        end_date: str
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        delay: float
            time.sleep(delay), default 1
        exists_skip: bool
            exists skip, by default False
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
        """

        if self.interval.lower() != "1D":
            logger.warning(f"currently supports 1D data updates: --interval 1D")

        # download qlib 1D data
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )

        # start/end date
        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        trading_date = (pd.Timestamp(str(calendar_df.iloc[-1, 0])) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # download data from vnstock
        # NOTE: when downloading data from vnstock, max_workers is recommended to be 1
        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length or 0)
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
            data_path=str(self.normalize_dir),
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # parse index
        _region = self.region.lower()
        if _region not in ["vn"]:
            logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
            return
        
        # For Vietnamese market, we could add index constituent parsing here
        # For now, we'll skip this step as it requires additional implementation
        logger.info("Index constituent parsing not yet implemented for Vietnamese market")


if __name__ == "__main__":
    fire.Fire(Run)
