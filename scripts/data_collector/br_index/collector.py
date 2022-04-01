# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
from pathlib import Path
import importlib
import datetime

import fire
import pandas as pd
from tqdm import tqdm
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase

YEAR_QUARTER = [
    "2003_1Q",
    "2003_2Q",
    "2003_3Q",
    "2004_1Q",
    "2004_2Q",
    "2004_3Q",
    "2005_1Q",
    "2005_2Q",
    "2005_3Q",
    "2006_1Q",
    "2006_2Q",
    "2006_3Q",
    "2007_1Q",
    "2007_2Q",
    "2007_3Q",
    "2008_1Q",
    "2008_2Q",
    "2008_3Q",
    "2009_1Q",
    "2009_2Q",
    "2009_3Q",
    "2010_1Q",
    "2010_2Q",
    "2010_3Q",
    "2011_1Q",
    "2011_2Q",
    "2011_3Q",
    "2012_1Q",
    "2012_2Q",
    "2012_3Q",
    "2013_1Q",
    "2013_2Q",
    "2013_3Q",
    "2014_1Q",
    "2014_2Q",
    "2014_3Q",
    "2015_1Q",
    "2015_2Q",
    "2015_3Q",
    "2016_1Q",
    "2016_2Q",
    "2016_3Q",
    "2017_1Q",
    "2017_2Q",
    "2017_3Q",
    "2018_1Q",
    "2018_2Q",
    "2018_3Q",
    "2019_1Q",
    "2019_2Q",
    "2019_3Q",
    "2020_1Q",
    "2020_2Q",
    "2020_3Q",
    "2021_1Q",
    "2021_2Q",
    "2021_3Q",
    "2022_1Q",
]

quarter_dict = {"1Q": "01-03", "2Q": "05-01", "3Q": "09-01"}


class IBOVIndex(IndexBase):

    ibov_index_composition = "https://raw.githubusercontent.com/igor17400/IBOV-HCI/main/historic_composition/{}.csv"

    def __init__(
        self,
        index_name: str,
        qlib_dir: [str, Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
    ):
        super(IBOVIndex, self).__init__(
            index_name=index_name, qlib_dir=qlib_dir, freq=freq, request_retry=request_retry, retry_sleep=retry_sleep
        )

        self.today = datetime.date.today()
        self.quarter = str(pd.Timestamp(self.today).quarter)
        self.year = str(self.today.year)

    @property
    def bench_start_date(self) -> pd.Timestamp:
        """
        The ibovespa index started on 2 January 1968 (wiki), however,
        no suitable data source that keeps track of ibovespa's history
        stocks composition has been found. Except from the repo indicated
        in README. Which keeps track of such information starting from
        the first quarter of 2003
        """
        return pd.Timestamp("2003-01-03")

    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """formatting the datetime in an instrument

        Parameters
        ----------
        inst_df: pd.DataFrame
            inst_df.columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        Returns
        -------
        inst_df: pd.DataFrame

        """
        logger.info("Formatting Datetime")
        if self.freq != "day":
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=23, minutes=59)).strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            inst_df[self.START_DATE_FIELD] = inst_df[self.START_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x)).strftime("%Y-%m-%d")
            )

            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x)).strftime("%Y-%m-%d")
            )
        return inst_df

    def format_quarter(self, cell: str):
        """
        Parameters
        ----------
        cell: str
            It must be on the format 2003_1Q --> year_quarter

        Returns
        ----------
        date: str
            Returns date in format 2003-03-01
        """
        cell_split = cell.split("_")
        return cell_split[0] + "-" + quarter_dict[cell_split[1]]

    def get_changes(self):
        """
        Access the index historic composition and compare it quarter
        by quarter and year by year in order to generate a file that
        keeps track of which stocks have been removed and which have
        been added.

        The Dataframe used as reference will provided the index
        composition for each year an quarter:
        pd.DataFrame:
            symbol
            SH600000
            SH600001
            .
            .
            .

        Parameters
        ----------
        self: is used to represent the instance of the class.

        Returns
        ----------
        pd.DataFrame:
            symbol      date        type
            SH600000  2019-11-11    add
            SH600001  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        """
        logger.info("Getting companies changes in {} index ...".format(self.index_name))

        try:
            df_changes_list = []
            for i in tqdm(range(len(YEAR_QUARTER) - 1)):
                df = pd.read_csv(self.ibov_index_composition.format(YEAR_QUARTER[i]), on_bad_lines="skip")["symbol"]
                df_ = pd.read_csv(self.ibov_index_composition.format(YEAR_QUARTER[i + 1]), on_bad_lines="skip")["symbol"]

                ## Remove Dataframe
                remove_date = YEAR_QUARTER[i].split("_")[0] + "-" + quarter_dict[YEAR_QUARTER[i].split("_")[1]]
                list_remove = list(df[~df.isin(df_)])
                df_removed = pd.DataFrame(
                    {
                        "date": len(list_remove) * [remove_date],
                        "type": len(list_remove) * ["remove"],
                        "symbol": list_remove,
                    }
                )

                ## Add Dataframe
                add_date = YEAR_QUARTER[i + 1].split("_")[0] + "-" + quarter_dict[YEAR_QUARTER[i + 1].split("_")[1]]
                list_add = list(df_[~df_.isin(df)])
                df_added = pd.DataFrame(
                    {"date": len(list_add) * [add_date], "type": len(list_add) * ["add"], "symbol": list_add}
                )

                df_changes_list.append(pd.concat([df_added, df_removed], sort=False))
                df = pd.concat(df_changes_list).reset_index(drop=True)
                df["symbol"] = df["symbol"].astype(str) + ".SA"

            return df

        except Exception as E:
            logger.error("An error occured while downloading 2008 index composition - {}".format(E))

    def get_new_companies(self):
        """
        Get latest index composition.
        The repo indicated on README has implemented a script
        to get the latest index composition from B3 website using
        selenium. Therefore, this method will download the file
        containing such composition

        Parameters
        ----------
        self: is used to represent the instance of the class.

        Returns
        ----------
        pd.DataFrame:
            symbol      start_date  end_date
            RRRP3	    2020-11-13	2022-03-02
            ALPA4	    2008-01-02	2022-03-02
            dtypes:
                symbol: str
                start_date: pd.Timestamp
                end_date: pd.Timestamp
        """
        logger.info("Getting new companies in {} index ...".format(self.index_name))

        try:
            ## Get index composition

            df_index = pd.read_csv(
                self.ibov_index_composition.format(self.year + "_" + self.quarter + "Q"), on_bad_lines="skip"
            )
            df_date_first_added = pd.read_csv(
                self.ibov_index_composition.format("date_first_added_" + self.year + "_" + self.quarter + "Q"),
                on_bad_lines="skip",
            )
            df = df_index.merge(df_date_first_added, on="symbol")[["symbol", "Date First Added"]]
            df[self.START_DATE_FIELD] = df["Date First Added"].map(self.format_quarter)

            # end_date will be our current quarter + 1, since the IBOV index updates itself every quarter
            df[self.END_DATE_FIELD] = self.year + "-" + quarter_dict[str(int(self.quarter) + 1) + "Q"]
            df = df[["symbol", self.START_DATE_FIELD, self.END_DATE_FIELD]]
            df["symbol"] = df["symbol"].astype(str) + ".SA"

            return df

        except Exception as E:
            logger.error("An error occured while getting new companies - {}".format(E))

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Código" in df.columns:
            return df.loc[:, ["Código"]].copy()


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
        index name, value from ["IBOV"]
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
        $ python collector.py --index_name IBOV --qlib_dir ~/.qlib/qlib_data/br_data --method parse_instruments

        # parse new companies
        $ python collector.py --index_name IBOV --qlib_dir ~/.qlib/qlib_data/br_data --method save_new_companies

    """
    _cur_module = importlib.import_module("data_collector.br_index.collector")
    obj = getattr(_cur_module, f"{index_name.upper()}Index")(
        qlib_dir=qlib_dir, index_name=index_name, freq=freq, request_retry=request_retry, retry_sleep=retry_sleep
    )
    getattr(obj, method)()


if __name__ == "__main__":
    fire.Fire(get_instruments)
