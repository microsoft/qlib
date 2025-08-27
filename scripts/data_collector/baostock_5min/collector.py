# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import sys
import copy
import fire
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Iterable, List

import qlib
from qlib.data import D

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import generate_minutes_calendar_from_daily, calc_adjusted_price


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
        return sorted({e[1] for e in hs300_stocks})

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = self.get_hs300_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol: str):
        return str(symbol).replace(".", "").upper()


class BaostockNormalizeHS3005min(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")

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
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")
        super(BaostockNormalizeHS3005min, self).__init__(date_field_name, symbol_field_name)

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].ffill()
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return self.generate_5min_from_daily(self.calendar_list_1d)

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
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan

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

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="5min",
            _1d_data_all=self.all_1d_data,
        )
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
        Changed the default value of: scripts.data_collector.base.BaseRun.
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
        """download data from Baostock

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

        Attention
        ---------
        qlib_data_1d_dir cannot be None, normalize 5min needs to use 1d data;

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
