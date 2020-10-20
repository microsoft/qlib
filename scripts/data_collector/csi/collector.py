# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import abc
import sys
import bisect
from io import BytesIO
from pathlib import Path

import fire
import requests
import pandas as pd
from lxml import etree
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.utils import get_hs_calendar_list as get_calendar_list


NEW_COMPANIES_URL = "http://www.csindex.com.cn/uploads/file/autofile/cons/{index_code}cons.xls"

INDEX_CHANGES_URL = "http://www.csindex.com.cn/zh-CN/search/total?key=%E5%85%B3%E4%BA%8E%E8%B0%83%E6%95%B4%E6%B2%AA%E6%B7%B1300%E5%92%8C%E4%B8%AD%E8%AF%81%E9%A6%99%E6%B8%AF100%E7%AD%89%E6%8C%87%E6%95%B0%E6%A0%B7%E6%9C%AC%E8%82%A1%E7%9A%84%E5%85%AC%E5%91%8A"


class CSIIndex:

    REMOVE = "remove"
    ADD = "add"

    def __init__(self, qlib_dir=None):
        """

        Parameters
        ----------
        qlib_dir: str
            qlib data dir, default "Path(__file__).parent/qlib_data"
        """

        if qlib_dir is None:
            qlib_dir = CUR_DIR.joinpath("qlib_data")
        self.instruments_dir = Path(qlib_dir).expanduser().resolve().joinpath("instruments")
        self.instruments_dir.mkdir(exist_ok=True, parents=True)
        self._calendar_list = None

        self.cache_dir = Path("~/.cache/csi").expanduser().resolve()
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    @property
    def calendar_list(self) -> list:
        """get history trading date

        Returns
        -------
        """
        return get_calendar_list(bench_code=self.index_name.upper())

    @property
    def new_companies_url(self):
        return NEW_COMPANIES_URL.format(index_code=self.index_code)

    @property
    def changes_url(self):
        return INDEX_CHANGES_URL

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def index_code(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def index_name(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def html_table_index(self):
        """Which table of changes in html

        CSI300: 0
        CSI100: 1
        :return:
        """
        raise NotImplementedError()

    def _get_trading_date_by_shift(self, trading_date: pd.Timestamp, shift=1):
        """get trading date by shift

        Parameters
        ----------
        shift : int
            shift, default is 1

        trading_date : pd.Timestamp
            trading date
        Returns
        -------

        """
        left_index = bisect.bisect_left(self.calendar_list, trading_date)
        try:
            res = self.calendar_list[left_index + shift]
        except IndexError:
            res = trading_date
        return res

    def _get_changes(self) -> pd.DataFrame:
        """get companies changes

        Returns
        -------

        """
        logger.info("get companies changes......")
        res = []
        for _url in self._get_change_notices_url():
            _df = self._read_change_from_url(_url)
            res.append(_df)
        logger.info("get companies changes finish")
        return pd.concat(res)

    @staticmethod
    def normalize_symbol(symbol):
        symbol = f"{int(symbol):06}"
        return f"SH{symbol}" if symbol.startswith("60") else f"SZ{symbol}"

    def _read_change_from_url(self, url: str) -> pd.DataFrame:
        """read change from url

        Parameters
        ----------
        url : str
            change url

        Returns
        -------

        """
        resp = requests.get(url)
        _text = resp.text

        date_list = re.findall(r"(\d{4}).*?年.*?(\d+).*?月.*?(\d+).*?日", _text)
        if len(date_list) >= 2:
            add_date = pd.Timestamp("-".join(date_list[0]))
        else:
            _date = pd.Timestamp("-".join(re.findall(r"(\d{4}).*?年.*?(\d+).*?月", _text)[0]))
            add_date = self._get_trading_date_by_shift(_date, shift=0)
        remove_date = self._get_trading_date_by_shift(add_date, shift=-1)
        logger.info(f"get {add_date} changes")
        try:
            excel_url = re.findall('.*href="(.*?xls.*?)".*', _text)[0]
            content = requests.get(f"http://www.csindex.com.cn{excel_url}").content
            _io = BytesIO(content)
            df_map = pd.read_excel(_io, sheet_name=None)
            with self.cache_dir.joinpath(
                f"{self.index_name.lower()}_changes_{add_date.strftime('%Y%m%d')}.{excel_url.split('.')[-1]}"
            ).open("wb") as fp:
                fp.write(content)
            tmp = []
            for _s_name, _type, _date in [("调入", self.ADD, add_date), ("调出", self.REMOVE, remove_date)]:
                _df = df_map[_s_name]
                _df = _df.loc[_df["指数代码"] == self.index_code, ["证券代码"]]
                _df = _df.applymap(self.normalize_symbol)
                _df.columns = ["symbol"]
                _df["type"] = _type
                _df["date"] = _date
                tmp.append(_df)
            df = pd.concat(tmp)
        except Exception:
            df = None
            _tmp_count = 0
            for _df in pd.read_html(resp.content):
                if _df.shape[-1] != 4:
                    continue
                _tmp_count += 1
                if self.html_table_index + 1 > _tmp_count:
                    continue
                tmp = []
                for _s, _type, _date in [
                    (_df.iloc[2:, 0], self.REMOVE, remove_date),
                    (_df.iloc[2:, 2], self.ADD, add_date),
                ]:
                    _tmp_df = pd.DataFrame()
                    _tmp_df["symbol"] = _s.map(self.normalize_symbol)
                    _tmp_df["type"] = _type
                    _tmp_df["date"] = _date
                    tmp.append(_tmp_df)
                df = pd.concat(tmp)
                df.to_csv(
                    str(
                        self.cache_dir.joinpath(
                            f"{self.index_name.lower()}_changes_{add_date.strftime('%Y%m%d')}.csv"
                        ).resolve()
                    )
                )
                break
        return df

    def _get_change_notices_url(self) -> list:
        """get change notices url

        Returns
        -------

        """
        resp = requests.get(self.changes_url)
        html = etree.HTML(resp.text)
        return html.xpath("//*[@id='itemContainer']//li/a/@href")

    def _get_new_companies(self):

        logger.info("get new companies")
        context = requests.get(self.new_companies_url).content
        with self.cache_dir.joinpath(
            f"{self.index_name.lower()}_new_companies.{self.new_companies_url.split('.')[-1]}"
        ).open("wb") as fp:
            fp.write(context)
        _io = BytesIO(context)
        df = pd.read_excel(_io)
        df = df.iloc[:, [0, 4]]
        df.columns = ["end_date", "symbol"]
        df["symbol"] = df["symbol"].map(self.normalize_symbol)
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["start_date"] = self.bench_start_date
        return df

    def parse_instruments(self):
        """parse csi300.txt

        Examples
        -------
            $ python collector.py parse_instruments --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        logger.info(f"start parse {self.index_name.lower()} companies.....")
        instruments_columns = ["symbol", "start_date", "end_date"]
        changers_df = self._get_changes()
        new_df = self._get_new_companies()
        logger.info("parse history companies by changes......")
        for _row in changers_df.sort_values("date", ascending=False).itertuples(index=False):
            if _row.type == self.ADD:
                min_end_date = new_df.loc[new_df["symbol"] == _row.symbol, "end_date"].min()
                new_df.loc[
                    (new_df["end_date"] == min_end_date) & (new_df["symbol"] == _row.symbol), "start_date"
                ] = _row.date
            else:
                _tmp_df = pd.DataFrame(
                    [[_row.symbol, self.bench_start_date, _row.date]], columns=["symbol", "start_date", "end_date"]
                )
                new_df = new_df.append(_tmp_df, sort=False)

        new_df.loc[:, instruments_columns].to_csv(
            self.instruments_dir.joinpath(f"{self.index_name.lower()}.txt"), sep="\t", index=False, header=None
        )
        logger.info(f"parse {self.index_name.lower()} companies finished.")


class CSI300(CSIIndex):
    @property
    def index_code(self):
        return "000300"

    @property
    def index_name(self):
        return "csi300"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2005-01-01")

    @property
    def html_table_index(self):
        return 0


class CSI100(CSIIndex):
    @property
    def index_code(self):
        return "000903"

    @property
    def index_name(self):
        return "csi100"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2006-05-29")

    @property
    def html_table_index(self):
        return 1


def parse_instruments(qlib_dir: str):
    """

    Parameters
    ----------
    qlib_dir: str
        qlib data dir, default "Path(__file__).parent/qlib_data"
    """
    qlib_dir = Path(qlib_dir).expanduser().resolve()
    qlib_dir.mkdir(exist_ok=True, parents=True)
    CSI300(qlib_dir).parse_instruments()
    CSI100(qlib_dir).parse_instruments()


if __name__ == "__main__":
    fire.Fire(parse_instruments)
