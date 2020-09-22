# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from lxml import etree
from loguru import logger
from yahooquery import Ticker

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from dump_bin import DumpData

SYMBOLS_URL = "http://app.finance.ifeng.com/hq/list.php?type=stock_a&class={s_type}"
CSI300_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20220101"


class YahooCollector:
    def __init__(self, save_dir: [str, Path], max_workers=4):

        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._stock_list = None
        self.max_workers = max_workers

    @property
    def stock_list(self):
        if self._stock_list is None:
            self._stock_list = self.get_stock_list()
        return self._stock_list

    @staticmethod
    def get_stock_list() -> list:
        _res = set()
        for _k, _v in (("ha", "ss"), ("sa", "sz"), ("gem", "sz")):
            resp = requests.get(SYMBOLS_URL.format(s_type=_k))
            _res |= set(
                map(
                    lambda x: "{}.{}".format(re.findall(r"\d+", x)[0], _v),
                    etree.HTML(resp.text).xpath("//div[@class='result']/ul//li/a/text()"),
                )
            )
        return sorted(list(_res))

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

    def collector_data(self):
        """collector data

        """
        logger.info("start collector yahoo data......")
        error_symbol = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as worker:
            futures = {}
            p_bar = tqdm(total=len(self.stock_list))
            for symbols in [
                self.stock_list[i : i + self.max_workers] for i in range(0, len(self.stock_list), self.max_workers)
            ]:
                resp = Ticker(symbols, asynchronous=True, max_workers=self.max_workers).history(period="max")
                if isinstance(resp, dict):
                    for symbol, df in resp.items():
                        if isinstance(df, pd.DataFrame):
                            futures[
                                worker.submit(
                                    self.save_stock, symbol, df.reset_index().rename(columns={"index": "date"})
                                )
                            ] = symbol
                        else:
                            error_symbol.append(symbol)
                else:
                    for symbol, df in resp.reset_index().groupby("symbol"):
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

        logger.info(error_symbol)
        logger.info(len(error_symbol))
        logger.info(len(self.stock_list))

        # TODO: from MSN
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
            df.sort_values("date", inplace=True)
            df.loc[df["volume"] <= 0, set(df.columns) - {"symbol", "date"}] = np.nan
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
            df.loc[:, columns + ["date"]].to_csv(self.normalize_dir.joinpath(file_path.name), index=False)

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
            df = df.loc[:, ["open", "close", "high", "low", "volume", "change", "factor"]]
            df.sort_values("date", inplace=True)
            df = df.set_index("date")
            df = df.loc[df.first_valid_index():]
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

    def download_data(self):
        """download data from Internet

        Examples
        ---------
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source

        """
        YahooCollector(self.source_dir, max_workers=self.max_workers).collector_data()

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
