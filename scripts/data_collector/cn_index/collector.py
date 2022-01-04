# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import abc
import sys
import importlib
from io import BytesIO
from typing import List, Iterable
from pathlib import Path

import fire
import requests
import pandas as pd
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase
from data_collector.utils import get_calendar_list, get_trading_date_by_shift, deco_retry


NEW_COMPANIES_URL = "https://csi-web-dev.oss-cn-shanghai-finance-1-pub.aliyuncs.com/static/html/csindex/public/uploads/file/autofile/cons/{index_code}cons.xls"


INDEX_CHANGES_URL = "https://www.csindex.com.cn/csindex-home/search/search-content?lang=cn&searchInput=%E5%85%B3%E4%BA%8E%E8%B0%83%E6%95%B4%E6%B2%AA%E6%B7%B1300%E5%92%8C%E4%B8%AD%E8%AF%81%E9%A6%99%E6%B8%AF100%E7%AD%89%E6%8C%87%E6%95%B0%E6%A0%B7%E6%9C%AC&pageNum={page_num}&pageSize={page_size}&sortField=date&dateRange=all&contentType=announcement"

REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36 Edg/91.0.864.48"
}


@deco_retry
def retry_request(url: str, method: str = "get", exclude_status: List = None):
    if exclude_status is None:
        exclude_status = []
    method_func = getattr(requests, method)
    _resp = method_func(url, headers=REQ_HEADERS)
    _status = _resp.status_code
    if _status not in exclude_status and _status != 200:
        raise ValueError(f"response status: {_status}, url={url}")
    return _resp


class CSIIndex(IndexBase):
    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        """get history trading date

        Returns
        -------
            calendar list
        """
        _calendar = getattr(self, "_calendar_list", None)
        if not _calendar:
            _calendar = get_calendar_list(bench_code=self.index_name.upper())
            setattr(self, "_calendar_list", _calendar)
        return _calendar

    @property
    def new_companies_url(self) -> str:
        return NEW_COMPANIES_URL.format(index_code=self.index_code)

    @property
    def changes_url(self) -> str:
        return INDEX_CHANGES_URL

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """
        Returns
        -------
            index start date
        """
        raise NotImplementedError("rewrite bench_start_date")

    @property
    @abc.abstractmethod
    def index_code(self) -> str:
        """
        Returns
        -------
            index code
        """
        raise NotImplementedError("rewrite index_code")

    @property
    @abc.abstractmethod
    def html_table_index(self) -> int:
        """Which table of changes in html

        CSI300: 0
        CSI100: 1
        :return:
        """
        raise NotImplementedError()

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
            inst_df[self.START_DATE_FIELD] = inst_df[self.START_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=9, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            )
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=15, minutes=0)).strftime("%Y-%m-%d %H:%M:%S")
            )
        return inst_df

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
        logger.info("get companies changes......")
        res = []
        for _url in self._get_change_notices_url():
            _df = self._read_change_from_url(_url)
            if not _df.empty:
                res.append(_df)
        logger.info("get companies changes finish")
        return pd.concat(res, sort=False)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """

        Parameters
        ----------
        symbol: str
            symbol

        Returns
        -------
            symbol
        """
        symbol = f"{int(symbol):06}"
        return f"SH{symbol}" if symbol.startswith("60") else f"SZ{symbol}"

    def _parse_excel(self, excel_url: str, add_date: pd.Timestamp, remove_date: pd.Timestamp) -> pd.DataFrame:
        content = retry_request(excel_url, exclude_status=[404]).content
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
            _df.columns = [self.SYMBOL_FIELD_NAME]
            _df["type"] = _type
            _df[self.DATE_FIELD_NAME] = _date
            tmp.append(_df)
        df = pd.concat(tmp)
        return df

    def _parse_table(self, content: str, add_date: pd.DataFrame, remove_date: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame()
        _tmp_count = 0
        for _df in pd.read_html(content):
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
                _tmp_df[self.SYMBOL_FIELD_NAME] = _s.map(self.normalize_symbol)
                _tmp_df["type"] = _type
                _tmp_df[self.DATE_FIELD_NAME] = _date
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

    def _read_change_from_url(self, url: str) -> pd.DataFrame:
        """read change from url

        Parameters
        ----------
        url : str
            change url

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
        resp = retry_request(url).json()["data"]
        title = resp["title"]
        if not title.startswith("关于"):
            return pd.DataFrame()
        if "沪深300" not in title:
            return pd.DataFrame()

        logger.info(f"load index data from https://www.csindex.com.cn/#/about/newsDetail?id={url.split('id=')[-1]}")
        _text = resp["content"]
        date_list = re.findall(r"(\d{4}).*?年.*?(\d+).*?月.*?(\d+).*?日", _text)
        if len(date_list) >= 2:
            add_date = pd.Timestamp("-".join(date_list[0]))
        else:
            _date = pd.Timestamp("-".join(re.findall(r"(\d{4}).*?年.*?(\d+).*?月", _text)[0]))
            add_date = get_trading_date_by_shift(self.calendar_list, _date, shift=0)
        if "盘后" in _text or "市后" in _text:
            add_date = get_trading_date_by_shift(self.calendar_list, add_date, shift=1)
        remove_date = get_trading_date_by_shift(self.calendar_list, add_date, shift=-1)

        excel_url = None
        if resp.get("enclosureList", []):
            excel_url = resp["enclosureList"][0]["fileUrl"]
        else:
            excel_url_list = re.findall('.*href="(.*?xls.*?)".*', _text)
            if excel_url_list:
                excel_url = excel_url_list[0]
                if not excel_url.startswith("http"):
                    excel_url = excel_url if excel_url.startswith("/") else "/" + excel_url
                    excel_url = f"http://www.csindex.com.cn{excel_url}"
        if excel_url:
            logger.info(f"get {add_date} changes from excel, title={title}, excel_url={excel_url}")
            try:
                df = self._parse_excel(excel_url, add_date, remove_date)
            except ValueError:
                logger.warning(f"error downloading file: {excel_url}, will parse the table from the content")
                df = self._parse_table(_text, add_date, remove_date)
        else:
            logger.info(f"get {add_date} changes from url content, title={title}")
            df = self._parse_table(_text, add_date, remove_date)
        return df

    def _get_change_notices_url(self) -> Iterable[str]:
        """get change notices url

        Returns
        -------
            [url1, url2]
        """
        page_num = 1
        page_size = 5
        data = retry_request(self.changes_url.format(page_size=page_size, page_num=page_num)).json()
        data = retry_request(self.changes_url.format(page_size=data["total"], page_num=page_num)).json()
        for item in data["data"]:
            yield f"https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById?id={item['id']}"

    def get_new_companies(self) -> pd.DataFrame:
        """

        Returns
        -------
            pd.DataFrame:

                symbol     start_date    end_date
                SH600000   2000-01-01    2099-12-31

            dtypes:
                symbol: str
                start_date: pd.Timestamp
                end_date: pd.Timestamp
        """
        logger.info("get new companies......")
        context = retry_request(self.new_companies_url).content
        with self.cache_dir.joinpath(
            f"{self.index_name.lower()}_new_companies.{self.new_companies_url.split('.')[-1]}"
        ).open("wb") as fp:
            fp.write(context)
        _io = BytesIO(context)
        df = pd.read_excel(_io)
        df = df.iloc[:, [0, 4]]
        df.columns = [self.END_DATE_FIELD, self.SYMBOL_FIELD_NAME]
        df[self.SYMBOL_FIELD_NAME] = df[self.SYMBOL_FIELD_NAME].map(self.normalize_symbol)
        df[self.END_DATE_FIELD] = pd.to_datetime(df[self.END_DATE_FIELD].astype(str))
        df[self.START_DATE_FIELD] = self.bench_start_date
        logger.info("end of get new companies.")
        return df


class CSI300(CSIIndex):
    @property
    def index_code(self):
        return "000300"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2005-01-01")

    @property
    def html_table_index(self):
        return 1


class CSI100(CSIIndex):
    @property
    def index_code(self):
        return "000903"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2006-05-29")

    @property
    def html_table_index(self):
        return 2


def get_instruments(
    qlib_dir: str,
    index_name: str,
    method: str = "parse_instruments",
    freq: str = "day",
    request_retry: int = 5,
    retry_sleep: int = 3,
):
    """

    Parameters
    ----------
    qlib_dir: str
        qlib data dir, default "Path(__file__).parent/qlib_data"
    index_name: str
        index name, value from ["csi100", "csi300"]
    method: str
        method, value from ["parse_instruments", "save_new_companies"]
    freq: str
        freq, value from ["day", "1min"]
    request_retry: int
        request retry, by default 5
    retry_sleep: int
        request sleep, by default 3

    Examples
    -------
        # parse instruments
        $ python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method parse_instruments

        # parse new companies
        $ python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method save_new_companies

    """
    _cur_module = importlib.import_module("data_collector.cn_index.collector")
    obj = getattr(_cur_module, f"{index_name.upper()}")(
        qlib_dir=qlib_dir, index_name=index_name, freq=freq, request_retry=request_retry, retry_sleep=retry_sleep
    )
    getattr(obj, method)()


if __name__ == "__main__":
    fire.Fire(get_instruments)
