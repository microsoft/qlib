# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from yahooquery import Ticker

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from dump_bin import DumpData
from data_collector.utils import get_hs_calendar_list as get_calendar_list, get_hs_stock_symbols

CSI300_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20220101"
MIN_NUMBERS_TRADING = 252 / 4


class YahooCollector:
    def __init__(self, save_dir: [str, Path], max_workers=4, asynchronous=False, max_collector_count=5, delay=0):

        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._delay = delay
        self._stock_list = None
        self.max_workers = max_workers
        self._asynchronous = asynchronous
        self._max_collector_count = max_collector_count
        self._mini_symbol_map = {}

    @property
    def stock_list(self):
        if self._stock_list is None:
            self._stock_list = get_hs_stock_symbols()
        return self._stock_list

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

        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        stock_path = self.save_dir.joinpath(f"{symbol}.csv")
        df["symbol"] = symbol
        df.to_csv(stock_path, index=False)

    def _temp_save_small_data(self, symbol, df):
        if len(df) <= MIN_NUMBERS_TRADING:
            logger.warning(f"the number of trading days of {symbol} is less than {MIN_NUMBERS_TRADING}!")
            _temp = self._mini_symbol_map.setdefault(symbol, [])
            _temp.append(df.copy())
        else:
            if symbol in self._mini_symbol_map:
                self._mini_symbol_map.pop(symbol)

    def _collector(self, stock_list):

        error_symbol = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as worker:
            futures = {}
            p_bar = tqdm(total=len(stock_list))
            for symbols in [stock_list[i : i + self.max_workers] for i in range(0, len(stock_list), self.max_workers)]:
                self._sleep()
                resp = Ticker(symbols, asynchronous=self._asynchronous, max_workers=self.max_workers).history(
                    period="max"
                )
                if isinstance(resp, dict):
                    for symbol, df in resp.items():
                        if isinstance(df, pd.DataFrame):
                            self._temp_save_small_data(self, df)
                            futures[
                                worker.submit(
                                    self.save_stock, symbol, df.reset_index().rename(columns={"index": "date"})
                                )
                            ] = symbol
                        else:
                            error_symbol.append(symbol)
                else:
                    for symbol, df in resp.reset_index().groupby("symbol"):
                        self._temp_save_small_data(self, df)
                        futures[worker.submit(self.save_stock, symbol, df)] = symbol
                p_bar.update(self.max_workers)
            p_bar.close()

            with tqdm(total=len(futures.values())) as p_bar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(e)
                        error_symbol.append(futures[future])
                    p_bar.update()
        print(error_symbol)
        logger.info(f"error symbol nums: {len(error_symbol)}")
        logger.info(f"current get symbol nums: {len(stock_list)}")
        error_symbol.extend(self._mini_symbol_map.keys())
        return error_symbol

    def collector_data(self):
        """collector data

        """
        logger.info("start collector yahoo data......")
        stock_list = self.stock_list
        for i in range(self._max_collector_count):
            if not stock_list:
                break
            logger.info(f"getting data: {i+1}")
            stock_list = self._collector(stock_list)
            logger.info(f"{i+1} finish.")
        for _symbol, _df_list in self._mini_symbol_map.items():
            self.save_stock(_symbol, max(_df_list, key=len))

        logger.warning(f"less than {MIN_NUMBERS_TRADING} stock list: {list(self._mini_symbol_map.keys())}")
        
        self.download_csi300_data()

    def download_csi300_data(self):
        # TODO: from MSN
        logger.info(f"get bench data: csi300(SH000300)......")
        df = pd.DataFrame(map(lambda x: x.split(","), requests.get(CSI300_BENCH_URL).json()["data"]["klines"]))
        df.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.astype(float, errors="ignore")
        df["adjclose"] = df["close"]
        df.to_csv(self.save_dir.joinpath("sh000300.csv"), index=False)


class Run:
    def __init__(self, source_dir=None, normalize_dir=None, qlib_dir=None, max_workers=4):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        qlib_dir: str
            qlib data dir; usage of provider_uri, default "Path(__file__).parent/qlib_data"
        max_workers: int
            Concurrent number, default is 4
        """
        if source_dir is None:
            source_dir = CUR_DIR.joinpath("source")
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.source_dir.mkdir(parents=True, exist_ok=True)

        if normalize_dir is None:
            normalize_dir = CUR_DIR.joinpath("normalize")
        self.normalize_dir = Path(normalize_dir).expanduser().resolve()
        self.normalize_dir.mkdir(parents=True, exist_ok=True)

        if qlib_dir is None:
            qlib_dir = CUR_DIR.joinpath("qlib_data")
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.qlib_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers

    def normalize_data(self):
        """normalize data

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize

        """

        def _normalize(file_path: Path):
            columns = ["open", "close", "high", "low", "volume"]
            df = pd.read_csv(file_path)
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="first")]

            # using China stock market data calendar
            df = df.reindex(pd.Index(get_calendar_list()))
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
            df.loc[:, columns].to_csv(self.normalize_dir.joinpath(file_path.name))

        with ThreadPoolExecutor(max_workers=self.max_workers) as worker:
            file_list = list(self.source_dir.glob("*.csv"))
            with tqdm(total=len(file_list)) as p_bar:
                for _ in worker.map(_normalize, file_list):
                    p_bar.update()

    def manual_adj_data(self):
        """manual adjust data

        Examples
        --------
            $ python collector.py manual_adj_data --normalize_dir ~/.qlib/stock_data/normalize

        """

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
            df.reset_index().to_csv(self.normalize_dir.joinpath(file_path.name), index=False)

        with ThreadPoolExecutor(max_workers=self.max_workers) as worker:
            file_list = list(self.normalize_dir.glob("*.csv"))
            with tqdm(total=len(file_list)) as p_bar:
                for _ in worker.map(_adj, file_list):
                    p_bar.update()

    def dump_data(self):
        """dump yahoo data

        Examples
        ---------
            $ python collector.py dump_data --normalize_dir ~/.qlib/stock_data/normalize_dir --qlib_dir ~/.qlib/stock_data/qlib_data

        """
        DumpData(csv_path=self.normalize_dir, qlib_dir=self.qlib_dir, works=self.max_workers).dump(
            include_fields="close,open,high,low,volume,change,factor"
        )

    def download_data(self, asynchronous=False, max_collector_count=5, delay=0):
        """download data from Internet

        Examples
        ---------
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source

        """
        YahooCollector(
            self.source_dir,
            max_workers=self.max_workers,
            asynchronous=asynchronous,
            max_collector_count=max_collector_count,
            delay=delay,
        ).collector_data()

    def download_csi300_data(self):
        YahooCollector(self.source_dir).download_csi300_data()

    def download_bench_data(self):
        """download bench stock data(SH000300)
        """

    def collector_data(self):
        """download -> normalize -> dump data

        Examples
        -------
            $ python collector.py collector_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize_dir --qlib_dir ~/.qlib/stock_data/qlib_data
        """
        self.download_data()
        self.normalize_data()
        self.manual_adj_data()
        self.dump_data()


if __name__ == "__main__":
    fire.Fire(Run)
