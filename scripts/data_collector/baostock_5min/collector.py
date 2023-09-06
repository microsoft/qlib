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
import baostock as bs
from abc import ABC
from pathlib import Path
from typing import Iterable

import fire
import numpy as np
import pandas as pd
from loguru import logger

import qlib
from qlib.data import D
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize

from data_collector.utils import generate_minutes_calendar_from_daily


class BaostockCollectorHS3005min(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="5min",
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
            freq, value from [5min], default 5min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        bs.login()
        interval="5min"
        super(BaostockCollectorHS3005min, self).__init__(
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
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        calendar_list = []
        while (rs.error_code == "0") & rs.next():
            calendar_list.append(rs.get_row_data())
        calendar_df = pd.DataFrame(calendar_list, columns=rs.fields)
        trade_calendar_df = calendar_df[~calendar_df["is_trading_day"].isin(["0"])]
        # bs.logout()
        return trade_calendar_df["calendar_date"].values

    @staticmethod
    def process_interval(interval: str):
        if interval == "1d":
            return {"interval": "d", "fields": "date,code,open,high,low,close,volume,amount,adjustflag"}
        if interval == "5min":
            return {"interval": "5", "fields": "date,time,code,open,high,low,close,volume,amount,adjustflag"}

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = self.get_data_from_remote(
            symbol=symbol, interval=interval, start_datetime=start_datetime, end_datetime=end_datetime
        )
        df.columns = ["date", "time", "symbol", "open", "high", "low", "close", "volume", "amount", "adjustflag"]
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
        df["date"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["date"] = df["date"].map(lambda x: pd.Timestamp(x) - pd.Timedelta(minutes=5))
        df.drop(["time"], axis=1, inplace=True)
        df["symbol"] = df["symbol"].map(lambda x: str(x).replace(".", "").upper())
        return df

    @staticmethod
    def get_data_from_remote(
        symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        rs = bs.query_history_k_data_plus(
            symbol,
            BaostockCollectorHS3005min.process_interval(interval=interval)["fields"],
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency=BaostockCollectorHS3005min.process_interval(interval=interval)["interval"],
            adjustflag="3",
        )
        if rs.error_code == "0" and len(rs.data) > 0:
            data_list = rs.data
            columns = rs.fields
            df = pd.DataFrame(data_list, columns=columns)
        return df

    def get_hs300_symbols(self) -> List[str]:
        hs300_stocks = []
        trade_calendar = self.get_trade_calendar()
        with tqdm(total=len(trade_calendar)) as p_bar:
            for date in trade_calendar:
                rs = bs.query_hs300_stocks(date=date)
                while rs.error_code == "0" and rs.next():
                    hs300_stocks.append(rs.get_row_data())
                p_bar.update()
        return sorted(set([e[1] for e in hs300_stocks]))

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = self.get_hs300_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol: str):
        return str(symbol).replace(".", "").upper()


class BaostockNormalizeHS3005min(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")
    # Whether the trading day of 5min data is consistent with 1d
    CONSISTENT_1d = True
    CALC_PAUSED_NUM = True

    def __init__(
        self, qlib_data_1d_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        qlib_data_1d_dir: str, Path
            the qlib data to be updated for yahoo, usually from: Normalised to 5min using local 1d data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        bs.login()
        qlib.init(provider_uri=qlib_data_1d_dir)
        # self.qlib_data_1d_dir = qlib_data_1d_dir
        super(BaostockNormalizeHS3005min, self).__init__(date_field_name, symbol_field_name)
        self._all_1d_data = self._get_all_1d_data()

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].fillna(method="ffill")
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # return list(D.calendar(freq="day"))
        return self.generate_5min_from_daily(self.calendar_list_1d)

    def _get_all_1d_data(self):
        df = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")
        df.reset_index(inplace=True)
        df.rename(columns={"datetime": self._date_field_name, "instrument": self._symbol_field_name}, inplace=True)
        df.columns = list(map(lambda x: x[1:] if x.startswith("$") else x, df.columns))
        return df

    @property
    def calendar_list_1d(self):
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = self._get_1d_calendar_list()
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        return calendar_list_1d

    @staticmethod
    def normalize_baostock(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(BaostockNormalizeHS3005min.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date() + pd.Timedelta(days=1)]
                .index
            )
        df.sort_index(inplace=True)
        # df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan

        change_series = BaostockNormalizeHS3005min.calc_change(df, last_close)
        # NOTE: The data obtained by Yahoo finance sometimes has exceptions
        # WARNING: If it is normal for a `symbol(exchange)` to differ by a factor of *89* to *111* for consecutive trading days,
        # WARNING: the logic in the following line needs to be modified
        _count = 0
        while True:
            # NOTE: may appear unusual for many days in a row
            change_series = BaostockNormalizeHS3005min.calc_change(df, last_close)
            _mask = (change_series >= 89) & (change_series <= 111)
            if not _mask.any():
                break
            _tmp_cols = ["high", "close", "low", "open"]
            df.loc[_mask, _tmp_cols] = df.loc[_mask, _tmp_cols] / 100
            _count += 1
            if _count >= 10:
                _symbol = df.loc[df[symbol_field_name].first_valid_index()]["symbol"]
                logger.warning(
                    f"{_symbol} `change` is abnormal for {_count} consecutive days, please check the specific data file carefully"
                )

        df["change"] = BaostockNormalizeHS3005min.calc_change(df, last_close)

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def generate_5min_from_daily(self, calendars: Iterable) -> pd.Index:
        return generate_minutes_calendar_from_daily(
            calendars, freq="5min", am_range=self.AM_RANGE, pm_range=self.PM_RANGE
        )

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

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: using daily data factor
        if df.empty:
            return df
        df = df.copy()
        df = df.sort_values(self._date_field_name)
        symbol = df.iloc[0][self._symbol_field_name]
        # get 1d data from baostock
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
                    self.generate_5min_from_daily(
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

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return list(D.calendar(freq="day"))

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_baostock(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # adjusted price
        df = self.adjusted_price(df)
        return df


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="5min", region="HS300"):
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
            freq, value from [5min, default 5min
        region: str
            region, value from ["HS300"], default "HS300"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"BaostockCollector{self.region.upper()}{self.interval}"

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
                hs300 5min, a week: 4 * 60 * 5

        Examples
        ---------
            # get hs300 5min data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --start 2022-01-01 --end 2022-01-30 --interval 5min --region HS300
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
            if interval==5min, qlib_data_1d_dir cannot be None, normalize 5min needs to use 1d data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn --version v3
                or:
                    download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --normalize_dir ~/.qlib/stock_data/source/hs300_5min_nor --region HS300 --interval 5min
        """
        if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
            raise ValueError(
                "If normalize 5min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
            )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )


if __name__ == "__main__":
    fire.Fire(Run)
