# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from yahooquery import Ticker
from dateutil.tz import tzlocal
from qlib.utils import code_to_fname

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.utils import get_calendar_list, get_hs_stock_symbols, get_us_stock_symbols

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"
REGION_CN = "CN"
REGION_US = "US"


class YahooCollector:
    START_DATETIME = pd.Timestamp("2000-01-01")
    HIGH_FREQ_START_DATETIME = pd.Timestamp(datetime.datetime.now() - pd.Timedelta(days=5 * 5))
    END_DATETIME = pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))

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
        show_1m_logging: bool = False,
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
            freq, value from [1m, 1d], default 1m
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: bool
            check data length, by default False
        limit_nums: int
            using for debug, by default None
        show_1m_logging: bool
            show 1m logging, by default False; if True, there may be many warning logs
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._delay = delay
        self._show_1m_logging = show_1m_logging
        self.stock_list = sorted(set(self.get_stock_list()))
        if limit_nums is not None:
            try:
                self.stock_list = self.stock_list[: int(limit_nums)]
            except Exception as e:
                logger.warning(f"Cannot use limit_nums={limit_nums}, the parameter will be ignored")
        self.max_workers = max_workers
        self._max_collector_count = max_collector_count
        self._mini_symbol_map = {}
        self._interval = interval
        self._check_small_data = check_data_length
        self._start_datetime = pd.Timestamp(str(start)) if start else self.START_DATETIME
        self._end_datetime = min(pd.Timestamp(str(end)) if end else self.END_DATETIME, self.END_DATETIME)
        if self._interval == "1m":
            self._start_datetime = max(self._start_datetime, self.HIGH_FREQ_START_DATETIME)
        elif self._interval == "1d":
            self._start_datetime = max(self._start_datetime, self.START_DATETIME)
        else:
            raise ValueError(f"interval error: {self._interval}")

        # using for 1m
        self._next_datetime = self.convert_datetime(self._start_datetime.date() + pd.Timedelta(days=1))
        self._latest_datetime = self.convert_datetime(self._end_datetime.date())

        self._start_datetime = self.convert_datetime(self._start_datetime)
        self._end_datetime = self.convert_datetime(self._end_datetime)

    @property
    @abc.abstractmethod
    def min_numbers_trading(self):
        # daily, one year: 252 / 4
        # us 1min, a week: 6.5 * 60 * 5
        # cn 1min, a week: 4 * 60 * 5
        raise NotImplementedError("rewrite min_numbers_trading")

    @abc.abstractmethod
    def get_stock_list(self):
        raise NotImplementedError("rewrite get_stock_list")

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    def convert_datetime(self, dt: [pd.Timestamp, datetime.date, str]):
        try:
            dt = pd.Timestamp(dt, tz=self._timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt

    def _sleep(self):
        time.sleep(self._delay)

    def save_stock(self, symbol, df: pd.DataFrame):
        """save stock data to file

        Parameters
        ----------
        symbol: str
            stock code
        df : pd.DataFrame
            df.columns must contain "symbol" and "datetime"
        """
        if df.empty:
            raise ValueError("df is empty")

        symbol = self.normalize_symbol(symbol)
        stock_path = self.save_dir.joinpath(f"{symbol}.csv")
        df["symbol"] = symbol
        if stock_path.exists():
            _temp_df = pd.read_csv(stock_path, nrows=0)
            df.loc[:, _temp_df.columns].to_csv(stock_path, index=False, header=False, mode="a")
        else:
            df.to_csv(stock_path, index=False, mode="w")

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

    def _get_from_remote(self, symbol):
        def _get_simple(start_, end_):
            self._sleep()
            error_msg = f"{symbol}-{self._interval}-{start_}-{end_}"

            def _show_logging_func():
                if self._interval == "1m" and self._show_1m_logging:
                    logger.warning(f"{error_msg}:{_resp}")

            try:
                _resp = Ticker(symbol, asynchronous=False).history(interval=self._interval, start=start_, end=end_)
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

        _result = None
        if self._interval == "1d":
            _result = _get_simple(self._start_datetime, self._end_datetime)
        elif self._interval == "1m":
            if self._next_datetime >= self._latest_datetime:
                _result = _get_simple(self._start_datetime, self._end_datetime)
            else:
                _res = []

                def _get_multi(start_, end_):
                    _resp = _get_simple(start_, end_)
                    if _resp is not None and not _resp.empty:
                        _res.append(_resp)

                for _s, _e in (
                    (self._start_datetime, self._next_datetime),
                    (self._latest_datetime, self._end_datetime),
                ):
                    _get_multi(_s, _e)
                for _start in pd.date_range(self._next_datetime, self._latest_datetime, closed="left"):
                    _end = _start + pd.Timedelta(days=1)
                    self._sleep()
                    _get_multi(_start, _end)
                if _res:
                    _result = pd.concat(_res, sort=False).sort_values(["symbol", "date"])
        else:
            raise ValueError(f"cannot support {self._interval}")
        return _result

    def _get_data(self, symbol):
        _result = None
        df = self._get_from_remote(symbol)
        if isinstance(df, pd.DataFrame):
            if not df.empty:
                if self._check_small_data:
                    if self._save_small_data(symbol, df) is not None:
                        _result = symbol
                        self.save_stock(symbol, df)
                else:
                    _result = symbol
                    self.save_stock(symbol, df)
        return _result

    def _collector(self, stock_list):

        error_symbol = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=len(stock_list)) as p_bar:
                for _symbol, _result in zip(stock_list, executor.map(self._get_data, stock_list)):
                    if _result is None:
                        error_symbol.append(_symbol)
                    p_bar.update()
        print(error_symbol)
        logger.info(f"error symbol nums: {len(error_symbol)}")
        logger.info(f"current get symbol nums: {len(stock_list)}")
        error_symbol.extend(self._mini_symbol_map.keys())
        return sorted(set(error_symbol))

    def collector_data(self):
        """collector data"""
        logger.info("start collector yahoo data......")
        stock_list = self.stock_list
        for i in range(self._max_collector_count):
            if not stock_list:
                break
            logger.info(f"getting data: {i+1}")
            stock_list = self._collector(stock_list)
            logger.info(f"{i+1} finish.")
        for _symbol, _df_list in self._mini_symbol_map.items():
            self.save_stock(_symbol, pd.concat(_df_list, sort=False).drop_duplicates(["date"]).sort_values(["date"]))
        if self._mini_symbol_map:
            logger.warning(f"less than {self.min_numbers_trading} stock list: {list(self._mini_symbol_map.keys())}")
        logger.info(f"total {len(self.stock_list)}, error: {len(set(stock_list))}")

        self.download_index_data()

    @abc.abstractmethod
    def download_index_data(self):
        """download index data"""
        raise NotImplementedError("rewrite download_index_data")

    @abc.abstractmethod
    def normalize_symbol(self, symbol: str):
        """normalize symbol"""
        raise NotImplementedError("rewrite normalize_symbol")


class YahooCollectorCN(YahooCollector):
    @property
    def min_numbers_trading(self):
        if self._interval == "1m":
            return 60 * 4 * 5
        elif self._interval == "1d":
            return 252 / 4

    def get_stock_list(self):
        logger.info("get HS stock symbos......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        # TODO: from MSN
        # FIXME: 1m
        if self._interval == "1d":
            _format = "%Y%m%d"
            _begin = self._start_datetime.strftime(_format)
            _end = (self._end_datetime + pd.Timedelta(days=-1)).strftime(_format)
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
                df.to_csv(self.save_dir.joinpath(f"sh{_index_code}.csv"), index=False)
        else:
            logger.warning(f"{self.__class__.__name__} {self._interval} does not support: downlaod_index_data")

    def normalize_symbol(self, symbol):
        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class YahooCollectorUS(YahooCollector):
    @property
    def min_numbers_trading(self):
        if self._interval == "1m":
            return 60 * 6.5 * 5
        elif self._interval == "1d":
            return 252 / 4

    def get_stock_list(self):
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


class YahooNormalize:
    COLUMNS = ["open", "close", "high", "low", "volume"]

    def __init__(self, source_dir: [str, Path], target_dir: [str, Path], max_workers: int = 16):
        """

        Parameters
        ----------
        source_dir: str or Path
            The directory where the raw data collected from the Internet is saved
        target_dir: str or Path
            Directory for normalize data
        max_workers: int
            Concurrent number, default is 16
        """
        if not (source_dir and target_dir):
            raise ValueError("source_dir and target_dir cannot be None")
        self._source_dir = Path(source_dir).expanduser()
        self._target_dir = Path(target_dir).expanduser()
        self._max_workers = max_workers
        self._calendar_list = self._get_calendar_list()

    def normalize_data(self):
        logger.info("normalize data......")

        def _normalize(source_path: Path):
            columns = copy.deepcopy(self.COLUMNS)
            df = pd.read_csv(source_path)
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="first")]
            if self._calendar_list is not None:
                df = df.reindex(pd.DataFrame(index=self._calendar_list).loc[df.index.min() : df.index.max()].index)
            df.sort_index(inplace=True)
            df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), set(df.columns) - {"symbol"}] = np.nan
            df["factor"] = df["adjclose"] / df["close"]
            for _col in columns:
                if _col == "volume":
                    df[_col] = df[_col] / df["factor"]
                else:
                    df[_col] = df[_col] * df["factor"]
            _tmp_series = df["close"].fillna(method="ffill")
            df["change"] = _tmp_series / _tmp_series.shift(1) - 1
            columns += ["change", "factor"]
            df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan
            df.index.names = ["date"]
            df.loc[:, columns].to_csv(self._target_dir.joinpath(source_path.name))

        with ThreadPoolExecutor(max_workers=self._max_workers) as worker:
            file_list = list(self._source_dir.glob("*.csv"))
            with tqdm(total=len(file_list)) as p_bar:
                for _ in worker.map(_normalize, file_list):
                    p_bar.update()

    def manual_adj_data(self):
        """adjust data"""
        logger.info("manual adjust data......")

        def _adj(file_path: Path):
            df = pd.read_csv(file_path)
            df = df.loc[:, ["open", "close", "high", "low", "volume", "change", "factor", "date"]]
            df.sort_values("date", inplace=True)
            df = df.set_index("date")
            df = df.loc[df.first_valid_index() :]
            _close = df["close"].iloc[0]
            for _col in df.columns:
                if _col == "volume":
                    df[_col] = df[_col] * _close
                elif _col != "change":
                    df[_col] = df[_col] / _close
                else:
                    pass
            df.reset_index().to_csv(self._target_dir.joinpath(file_path.name), index=False)

        with ThreadPoolExecutor(max_workers=self._max_workers) as worker:
            file_list = list(self._target_dir.glob("*.csv"))
            with tqdm(total=len(file_list)) as p_bar:
                for _ in worker.map(_adj, file_list):
                    p_bar.update()

    def normalize(self):
        self.normalize_data()
        self.manual_adj_data()

    @abc.abstractmethod
    def _get_calendar_list(self):
        """Get benchmark calendar"""
        raise NotImplementedError("")


class YahooNormalizeUS(YahooNormalize):
    def _get_calendar_list(self):
        # TODO: from MSN
        return get_calendar_list("US_ALL")


class YahooNormalizeCN(YahooNormalize):
    def _get_calendar_list(self):
        # TODO: from MSN
        return get_calendar_list("ALL")


class Run:
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=4, region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 4
        region: str
            region, value from ["CN", "US"], default "CN"
        """
        if source_dir is None:
            source_dir = CUR_DIR.joinpath("source")
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.source_dir.mkdir(parents=True, exist_ok=True)

        if normalize_dir is None:
            normalize_dir = CUR_DIR.joinpath("normalize")
        self.normalize_dir = Path(normalize_dir).expanduser().resolve()
        self.normalize_dir.mkdir(parents=True, exist_ok=True)

        self._cur_module = importlib.import_module("collector")
        self.max_workers = max_workers
        self.region = region

    def download_data(
        self,
        max_collector_count=5,
        delay=0,
        start=None,
        end=None,
        interval="1d",
        check_data_length=False,
        limit_nums=None,
        show_1m_logging=False,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 5
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1m, 1d], default 1m
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``
        check_data_length: bool
            check data length, by default False
        limit_nums: int
            using for debug, by default None
        show_1m_logging: bool
            show 1m logging, by default False; if True, there may be many warning logs

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """

        _class = getattr(self._cur_module, f"YahooCollector{self.region.upper()}")
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
            show_1m_logging=show_1m_logging,
        ).collector_data()

    def normalize_data(self):
        """normalize data

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN
        """
        _class = getattr(self._cur_module, f"YahooNormalize{self.region.upper()}")
        _class(self.source_dir, self.normalize_dir, self.max_workers).normalize()

    def collector_data(
        self,
        max_collector_count=5,
        delay=0,
        start=None,
        end=None,
        interval="1d",
        check_data_length=False,
        limit_nums=None,
        show_1m_logging=False,
    ):
        """download -> normalize

        Parameters
        ----------
        max_collector_count: int
            default 5
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1m, 1d], default 1m
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``
        check_data_length: bool
            check data length, by default False
        limit_nums: int
            using for debug, by default None
        show_1m_logging: bool
            show 1m logging, by default False; if True, there may be many warning logs

        Examples
        -------
        python collector.py collector_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
        """
        self.download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            show_1m_logging=show_1m_logging,
        )
        self.normalize_data()


if __name__ == "__main__":
    fire.Fire(Run)
