# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import abc
import sys
import datetime
from io import BytesIO
from typing import List, Iterable
from pathlib import Path

import fire
import requests
import pandas as pd
import baostock as bs
from tqdm import tqdm
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase
from data_collector.utils import get_calendar_list, get_trading_date_by_shift, deco_retry
from data_collector.utils import get_instruments


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
    def html_table_index(self) -> int:
        """Which table of changes in html

        CSI300: 0
        CSI100: 1
        :return:
        """
        raise NotImplementedError("rewrite html_table_index")

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
        return f"SH{symbol}" if symbol.startswith("60") or symbol.startswith("688") else f"SZ{symbol}"

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
            if _df.shape[-1] != 4 or _df.isnull().loc(0)[0][0]:
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
        The parameter url is from the _get_change_notices_url method.
        Determine the stock add_date/remove_date based on the title.
        The response contains three cases:
            1.Only excel_url(extract data from excel_url)
            2.Both the excel_url and the body text(try to extract data from excel_url first, and then try to extract data from body text)
            3.Only body text(extract data from body text)

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
            try:
                logger.info(f"get {add_date} changes from the excel, title={title}, excel_url={excel_url}")
                df = self._parse_excel(excel_url, add_date, remove_date)
            except ValueError:
                logger.info(
                    f"get {add_date} changes from the web page, title={title}, url=https://www.csindex.com.cn/#/about/newsDetail?id={url.split('id=')[-1]}"
                )
                df = self._parse_table(_text, add_date, remove_date)
        else:
            logger.info(
                f"get {add_date} changes from the web page, title={title}, url=https://www.csindex.com.cn/#/about/newsDetail?id={url.split('id=')[-1]}"
            )
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


class CSI300Index(CSIIndex):
    @property
    def index_code(self):
        return "000300"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2005-01-01")

    @property
    def html_table_index(self) -> int:
        return 0


class CSI100Index(CSIIndex):
    @property
    def index_code(self):
        return "000903"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2006-05-29")

    @property
    def html_table_index(self) -> int:
        return 1


class CSI500Index(CSIIndex):
    @property
    def index_code(self) -> str:
        return "000905"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2007-01-15")

    def get_changes(self) -> pd.DataFrame:
        """get companies changes

        Return
        --------
           pd.DataFrame:
               symbol      date        type
               SH600000  2019-11-11    add
               SH600000  2020-11-10    remove
           dtypes:
               symbol: str
               date: pd.Timestamp
               type: str, value from ["add", "remove"]
        """
        return self.get_changes_with_history_companies(self.get_history_companies())

    def get_history_companies(self) -> pd.DataFrame:
        """

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
        bs.login()
        today = pd.Timestamp.now()
        date_range = pd.DataFrame(pd.date_range(start="2007-01-15", end=today, freq="7D"))[0].dt.date
        ret_list = []
        col = ["date", "symbol", "code_name"]
        for date in tqdm(date_range, desc="Download CSI500"):
            rs = bs.query_zz500_stocks(date=str(date))
            zz500_stocks = []
            while (rs.error_code == "0") & rs.next():
                zz500_stocks.append(rs.get_row_data())
            result = pd.DataFrame(zz500_stocks, columns=col)
            result["symbol"] = result["symbol"].apply(lambda x: x.replace(".", "").upper())
            result = self.get_data_from_baostock(date)
            ret_list.append(result[["date", "symbol"]])
        bs.logout()
        return pd.concat(ret_list, sort=False)

    @staticmethod
    def get_data_from_baostock(date) -> pd.DataFrame:
        """
        Data source: http://baostock.com/baostock/index.php/%E4%B8%AD%E8%AF%81500%E6%88%90%E5%88%86%E8%82%A1
        Avoid a large number of parallel data acquisition,
        such as 1000 times of concurrent data acquisition, because IP will be blocked

        Returns
        -------
            pd.DataFrame:
                date      symbol        code_name
                SH600039  2007-01-15    四川路桥
                SH600051  2020-01-15    宁波联合
            dtypes:
                date: pd.Timestamp
                symbol: str
                code_name: str
        """
        col = ["date", "symbol", "code_name"]
        rs = bs.query_zz500_stocks(date=str(date))
        zz500_stocks = []
        while (rs.error_code == "0") & rs.next():
            zz500_stocks.append(rs.get_row_data())
        result = pd.DataFrame(zz500_stocks, columns=col)
        result["symbol"] = result["symbol"].apply(lambda x: x.replace(".", "").upper())
        return result

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
        today = pd.Timestamp.now().normalize()
        bs.login()
        result = self.get_data_from_baostock(today.strftime("%Y-%m-%d"))
        bs.logout()
        df = result[["date", "symbol"]]
        df.columns = [self.END_DATE_FIELD, self.SYMBOL_FIELD_NAME]
        df[self.END_DATE_FIELD] = today
        df[self.START_DATE_FIELD] = self.bench_start_date
        logger.info("end of get new companies.")
        return df


if __name__ == "__main__":
    fire.Fire(get_instruments)
