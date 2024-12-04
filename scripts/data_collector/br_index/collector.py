# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from functools import partial
import sys
from pathlib import Path
import datetime

import fire
import pandas as pd
from tqdm import tqdm
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase
from data_collector.utils import get_instruments

quarter_dict = {"1Q": "01-03", "2Q": "05-01", "3Q": "09-01"}


class IBOVIndex(IndexBase):
    ibov_index_composition = "https://raw.githubusercontent.com/igor17400/IBOV-HCI/main/historic_composition/{}.csv"
    years_4_month_periods = []

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

        self.today: datetime = datetime.date.today()
        self.current_4_month_period = self.get_current_4_month_period(self.today.month)
        self.year = str(self.today.year)
        self.years_4_month_periods = self.get_four_month_period()

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

    def get_current_4_month_period(self, current_month: int):
        """
        This function is used to calculated what is the current
        four month period for the current month. For example,
        If the current month is August 8, its four month period
        is 2Q.

        OBS: In english Q is used to represent *quarter*
        which means a three month period. However, in
        portuguese we use Q to represent a four month period.
        In other words,

        Jan, Feb, Mar, Apr: 1Q
        May, Jun, Jul, Aug: 2Q
        Sep, Oct, Nov, Dez: 3Q

        Parameters
        ----------
        month : int
            Current month (1 <= month <= 12)

        Returns
        -------
        current_4m_period:str
            Current Four Month Period (1Q or 2Q or 3Q)
        """
        if current_month < 5:
            return "1Q"
        if current_month < 9:
            return "2Q"
        if current_month <= 12:
            return "3Q"
        else:
            return -1

    def get_four_month_period(self):
        """
        The ibovespa index is updated every four months.
        Therefore, we will represent each time period as 2003_1Q
        which means 2003 first four mount period (Jan, Feb, Mar, Apr)
        """
        four_months_period = ["1Q", "2Q", "3Q"]
        init_year = 2003
        now = datetime.datetime.now()
        current_year = now.year
        current_month = now.month
        for year in [item for item in range(init_year, current_year)]:  # pylint: disable=R1721
            for el in four_months_period:
                self.years_4_month_periods.append(str(year) + "_" + el)
        # For current year the logic must be a little different
        current_4_month_period = self.get_current_4_month_period(current_month)
        for i in range(int(current_4_month_period[0])):
            self.years_4_month_periods.append(str(current_year) + "_" + str(i + 1) + "Q")
        return self.years_4_month_periods

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
            It must be on the format 2003_1Q --> years_4_month_periods

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
            for i in tqdm(range(len(self.years_4_month_periods) - 1)):
                df = pd.read_csv(
                    self.ibov_index_composition.format(self.years_4_month_periods[i]), on_bad_lines="skip"
                )["symbol"]
                df_ = pd.read_csv(
                    self.ibov_index_composition.format(self.years_4_month_periods[i + 1]), on_bad_lines="skip"
                )["symbol"]

                ## Remove Dataframe
                remove_date = (
                    self.years_4_month_periods[i].split("_")[0]
                    + "-"
                    + quarter_dict[self.years_4_month_periods[i].split("_")[1]]
                )
                list_remove = list(df[~df.isin(df_)])
                df_removed = pd.DataFrame(
                    {
                        "date": len(list_remove) * [remove_date],
                        "type": len(list_remove) * ["remove"],
                        "symbol": list_remove,
                    }
                )

                ## Add Dataframe
                add_date = (
                    self.years_4_month_periods[i + 1].split("_")[0]
                    + "-"
                    + quarter_dict[self.years_4_month_periods[i + 1].split("_")[1]]
                )
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
                self.ibov_index_composition.format(self.year + "_" + self.current_4_month_period), on_bad_lines="skip"
            )
            df_date_first_added = pd.read_csv(
                self.ibov_index_composition.format("date_first_added_" + self.year + "_" + self.current_4_month_period),
                on_bad_lines="skip",
            )
            df = df_index.merge(df_date_first_added, on="symbol")[["symbol", "Date First Added"]]
            df[self.START_DATE_FIELD] = df["Date First Added"].map(self.format_quarter)

            # end_date will be our current quarter + 1, since the IBOV index updates itself every quarter
            df[self.END_DATE_FIELD] = self.year + "-" + quarter_dict[self.current_4_month_period]
            df = df[["symbol", self.START_DATE_FIELD, self.END_DATE_FIELD]]
            df["symbol"] = df["symbol"].astype(str) + ".SA"

            return df

        except Exception as E:
            logger.error("An error occured while getting new companies - {}".format(E))

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Código" in df.columns:
            return df.loc[:, ["Código"]].copy()


if __name__ == "__main__":
    fire.Fire(partial(get_instruments, market_index="br_index"))
