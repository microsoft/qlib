# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from functools import partial
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

import fire
import requests
import pandas as pd
from tqdm import tqdm
from loguru import logger


CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase
from data_collector.utils import deco_retry, get_calendar_list, get_trading_date_by_shift
from data_collector.utils import get_instruments


WIKI_URL = "https://en.wikipedia.org/wiki"

WIKI_INDEX_NAME_MAP = {
    "NASDAQ100": "NASDAQ-100",
    "SP500": "List_of_S%26P_500_companies",
    "SP400": "List_of_S%26P_400_companies",
    "DJIA": "Dow_Jones_Industrial_Average",
}


class WIKIIndex(IndexBase):
    # NOTE: The US stock code contains "PRN", and the directory cannot be created on Windows system, use the "_" prefix
    # https://superuser.com/questions/613313/why-cant-we-make-con-prn-null-folder-in-windows
    INST_PREFIX = ""

    def __init__(
        self,
        index_name: str,
        qlib_dir: [str, Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
    ):
        super(WIKIIndex, self).__init__(
            index_name=index_name, qlib_dir=qlib_dir, freq=freq, request_retry=request_retry, retry_sleep=retry_sleep
        )

        self._target_url = f"{WIKI_URL}/{WIKI_INDEX_NAME_MAP[self.index_name.upper()]}"

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """
        Returns
        -------
            index start date
        """
        raise NotImplementedError("rewrite bench_start_date")

    @abc.abstractmethod
    def get_changes(self) -> pd.DataFrame:
        """get companies changes

        Returns
        -------
            pd.DataFrame:
                symbol      date        type
                SH600000  2019-11-11    add
                SH600000  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        """
        raise NotImplementedError("rewrite get_changes")

    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """formatting the datetime in an instrument

        Parameters
        ----------
        inst_df: pd.DataFrame
            inst_df.columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        Returns
        -------

        """
        if self.freq != "day":
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=23, minutes=59)).strftime("%Y-%m-%d %H:%M:%S")
            )
        return inst_df

    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        """get history trading date

        Returns
        -------
            calendar list
        """
        _calendar_list = getattr(self, "_calendar_list", None)
        if _calendar_list is None:
            _calendar_list = list(filter(lambda x: x >= self.bench_start_date, get_calendar_list("US_ALL")))
            setattr(self, "_calendar_list", _calendar_list)
        return _calendar_list

    def _request_new_companies(self) -> requests.Response:
        resp = requests.get(self._target_url, timeout=None)
        if resp.status_code != 200:
            raise ValueError(f"request error: {self._target_url}")

        return resp

    def set_default_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        _df = df.copy()
        _df[self.SYMBOL_FIELD_NAME] = _df[self.SYMBOL_FIELD_NAME].str.strip()
        _df[self.START_DATE_FIELD] = self.bench_start_date
        _df[self.END_DATE_FIELD] = self.DEFAULT_END_DATE
        return _df.loc[:, self.INSTRUMENTS_COLUMNS]

    def get_new_companies(self):
        logger.info(f"get new companies {self.index_name} ......")
        _data = deco_retry(retry=self._request_retry, retry_sleep=self._retry_sleep)(self._request_new_companies)()
        df_list = pd.read_html(_data.text)
        for _df in df_list:
            _df = self.filter_df(_df)
            if (_df is not None) and (not _df.empty):
                _df.columns = [self.SYMBOL_FIELD_NAME]
                _df = self.set_default_date_range(_df)
                logger.info(f"end of get new companies {self.index_name} ......")
                return _df

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("rewrite filter_df")


class NASDAQ100Index(WIKIIndex):
    HISTORY_COMPANIES_URL = (
        "https://indexes.nasdaqomx.com/Index/WeightingData?id=NDX&tradeDate={trade_date}T00%3A00%3A00.000&timeOfDay=SOD"
    )
    MAX_WORKERS = 16

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) >= 100 and "Ticker" in df.columns:
            return df.loc[:, ["Ticker"]].copy()

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2003-01-02")

    @deco_retry
    def _request_history_companies(self, trade_date: pd.Timestamp, use_cache: bool = True) -> pd.DataFrame:
        trade_date = trade_date.strftime("%Y-%m-%d")
        cache_path = self.cache_dir.joinpath(f"{trade_date}_history_companies.pkl")
        if cache_path.exists() and use_cache:
            df = pd.read_pickle(cache_path)
        else:
            url = self.HISTORY_COMPANIES_URL.format(trade_date=trade_date)
            resp = requests.post(url, timeout=None)
            if resp.status_code != 200:
                raise ValueError(f"request error: {url}")
            df = pd.DataFrame(resp.json()["aaData"])
            df[self.DATE_FIELD_NAME] = trade_date
            df.rename(columns={"Name": "name", "Symbol": self.SYMBOL_FIELD_NAME}, inplace=True)
            if not df.empty:
                df.to_pickle(cache_path)
        return df

    def get_history_companies(self):
        logger.info(f"start get history companies......")
        all_history = []
        error_list = []
        with tqdm(total=len(self.calendar_list)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                for _trading_date, _df in zip(
                    self.calendar_list, executor.map(self._request_history_companies, self.calendar_list)
                ):
                    if _df.empty:
                        error_list.append(_trading_date)
                    else:
                        all_history.append(_df)
                    p_bar.update()

        if error_list:
            logger.warning(f"get error: {error_list}")
        logger.info(f"total {len(self.calendar_list)}, error {len(error_list)}")
        logger.info(f"end of get history companies.")
        return pd.concat(all_history, sort=False)

    def get_changes(self):
        return self.get_changes_with_history_companies(self.get_history_companies())


class DJIAIndex(WIKIIndex):
    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2000-01-01")

    def get_changes(self) -> pd.DataFrame:
        pass

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Symbol" in df.columns:
            _df = df.loc[:, ["Symbol"]].copy()
            _df["Symbol"] = _df["Symbol"].apply(lambda x: x.split(":")[-1])
            return _df

    def parse_instruments(self):
        logger.warning(f"No suitable data source has been found!")


class SP500Index(WIKIIndex):
    WIKISP500_CHANGES_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("1999-01-01")

    def get_changes(self) -> pd.DataFrame:
        logger.info(f"get sp500 history changes......")
        # NOTE: may update the index of the table
        changes_df = pd.read_html(self.WIKISP500_CHANGES_URL)[-1]
        changes_df = changes_df.iloc[:, [0, 1, 3]]
        changes_df.columns = [self.DATE_FIELD_NAME, self.ADD, self.REMOVE]
        changes_df[self.DATE_FIELD_NAME] = pd.to_datetime(changes_df[self.DATE_FIELD_NAME])
        _result = []
        for _type in [self.ADD, self.REMOVE]:
            _df = changes_df.copy()
            _df[self.CHANGE_TYPE_FIELD] = _type
            _df[self.SYMBOL_FIELD_NAME] = _df[_type]
            _df.dropna(subset=[self.SYMBOL_FIELD_NAME], inplace=True)
            if _type == self.ADD:
                _df[self.DATE_FIELD_NAME] = _df[self.DATE_FIELD_NAME].apply(
                    lambda x: get_trading_date_by_shift(self.calendar_list, x, 0)
                )
            else:
                _df[self.DATE_FIELD_NAME] = _df[self.DATE_FIELD_NAME].apply(
                    lambda x: get_trading_date_by_shift(self.calendar_list, x, -1)
                )
            _result.append(_df[[self.DATE_FIELD_NAME, self.CHANGE_TYPE_FIELD, self.SYMBOL_FIELD_NAME]])
        logger.info(f"end of get sp500 history changes.")
        return pd.concat(_result, sort=False)

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Symbol" in df.columns:
            return df.loc[:, ["Symbol"]].copy()


class SP400Index(WIKIIndex):
    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2000-01-01")

    def get_changes(self) -> pd.DataFrame:
        pass

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Ticker symbol" in df.columns:
            return df.loc[:, ["Ticker symbol"]].copy()

    def parse_instruments(self):
        logger.warning(f"No suitable data source has been found!")


if __name__ == "__main__":
    fire.Fire(partial(get_instruments, market_index="us_index"))
