# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
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


NEW_COMPANIES_URL = "http://www.csindex.com.cn/uploads/file/autofile/cons/000300cons.xls"

CSI300_CHANGES_URL = "http://www.csindex.com.cn/zh-CN/search/total?key=%E5%85%B3%E4%BA%8E%E8%B0%83%E6%95%B4%E6%B2%AA%E6%B7%B1300%E5%92%8C%E4%B8%AD%E8%AF%81%E9%A6%99%E6%B8%AF100%E7%AD%89%E6%8C%87%E6%95%B0%E6%A0%B7%E6%9C%AC%E8%82%A1%E7%9A%84%E5%85%AC%E5%91%8A"

CSI300_START_DATE = pd.Timestamp("2005-01-01")


class CSI300:

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

    @property
    def calendar_list(self) -> list:
        """get history trading date

        Returns
        -------
        """
        return get_calendar_list(bench=True)

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
            _io = BytesIO(requests.get(f"http://www.csindex.com.cn{excel_url}").content)
            df_map = pd.read_excel(_io, sheet_name=None)
            tmp = []
            for _s_name, _type, _date in [("调入", self.ADD, add_date), ("调出", self.REMOVE, remove_date)]:
                _df = df_map[_s_name]
                _df = _df.loc[_df["指数代码"] == "000300", ["证券代码"]]
                _df = _df.applymap(self.normalize_symbol)
                _df.columns = ["symbol"]
                _df["type"] = _type
                _df["date"] = _date
                tmp.append(_df)
            df = pd.concat(tmp)
        except Exception:
            df = None
            for _df in pd.read_html(resp.content):
                if _df.shape[-1] != 4:
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
                break
        return df

    @staticmethod
    def _get_change_notices_url() -> list:
        """get change notices url

        Returns
        -------

        """
        resp = requests.get(CSI300_CHANGES_URL)
        html = etree.HTML(resp.text)
        return html.xpath("//*[@id='itemContainer']//li/a/@href")

    def _get_new_companies(self):

        logger.info("get new companies")
        _io = BytesIO(requests.get(NEW_COMPANIES_URL).content)
        df = pd.read_excel(_io)
        df = df.iloc[:, [0, 4]]
        df.columns = ["end_date", "symbol"]
        df["symbol"] = df["symbol"].map(self.normalize_symbol)
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["start_date"] = CSI300_START_DATE
        return df

    def parse_instruments(self):
        """parse csi300.txt

        Examples
        -------
            $ python collector.py parse_instruments --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        logger.info("start parse csi300 companies.....")
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
                    [[_row.symbol, CSI300_START_DATE, _row.date]], columns=["symbol", "start_date", "end_date"]
                )
                new_df = new_df.append(_tmp_df, sort=False)

        new_df.loc[:, instruments_columns].to_csv(
            self.instruments_dir.joinpath("csi300.txt"), sep="\t", index=False, header=None
        )
        logger.info("parse csi300 companies finished.")


if __name__ == "__main__":
    fire.Fire(CSI300)
