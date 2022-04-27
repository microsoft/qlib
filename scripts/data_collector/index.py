import sys
import abc
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent))


from data_collector.utils import get_trading_date_by_shift


class IndexBase:
    DEFAULT_END_DATE = pd.Timestamp("2099-12-31")
    SYMBOL_FIELD_NAME = "symbol"
    DATE_FIELD_NAME = "date"
    START_DATE_FIELD = "start_date"
    END_DATE_FIELD = "end_date"
    CHANGE_TYPE_FIELD = "type"
    INSTRUMENTS_COLUMNS = [SYMBOL_FIELD_NAME, START_DATE_FIELD, END_DATE_FIELD]
    REMOVE = "remove"
    ADD = "add"
    INST_PREFIX = ""

    def __init__(
        self,
        index_name: str,
        qlib_dir: [str, Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
    ):
        """

        Parameters
        ----------
        index_name: str
            index name
        qlib_dir: str
            qlib directory, by default Path(__file__).resolve().parent.joinpath("qlib_data")
        freq: str
            freq, value from ["day", "1min"]
        request_retry: int
            request retry, by default 5
        retry_sleep: int
            request sleep, by default 3
        """
        self.index_name = index_name
        if qlib_dir is None:
            qlib_dir = Path(__file__).resolve().parent.joinpath("qlib_data")
        self.instruments_dir = Path(qlib_dir).expanduser().resolve().joinpath("instruments")
        self.instruments_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = Path(f"~/.cache/qlib/index/{self.index_name}").expanduser().resolve()
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._request_retry = request_retry
        self._retry_sleep = retry_sleep
        self.freq = freq

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
    def calendar_list(self) -> List[pd.Timestamp]:
        """get history trading date

        Returns
        -------
            calendar list
        """
        raise NotImplementedError("rewrite calendar_list")

    @abc.abstractmethod
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
        raise NotImplementedError("rewrite get_new_companies")

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

    @abc.abstractmethod
    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """formatting the datetime in an instrument

        Parameters
        ----------
        inst_df: pd.DataFrame
            inst_df.columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        Returns
        -------

        """
        raise NotImplementedError("rewrite format_datetime")

    def save_new_companies(self):
        """save new companies

        Examples
        -------
            $ python collector.py save_new_companies --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        df = self.get_new_companies()
        if df is None or df.empty:
            raise ValueError(f"get new companies error: {self.index_name}")
        df = df.drop_duplicates([self.SYMBOL_FIELD_NAME])
        df.loc[:, self.INSTRUMENTS_COLUMNS].to_csv(
            self.instruments_dir.joinpath(f"{self.index_name.lower()}_only_new.txt"), sep="\t", index=False, header=None
        )

    def get_changes_with_history_companies(self, history_companies: pd.DataFrame) -> pd.DataFrame:
        """get changes with history companies

        Parameters
        ----------
        history_companies : pd.DataFrame
            symbol        date
            SH600000   2020-11-11

            dtypes:
                symbol: str
                date: pd.Timestamp

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
        logger.info("parse changes from history companies......")
        last_code = []
        result_df_list = []
        _columns = [self.DATE_FIELD_NAME, self.SYMBOL_FIELD_NAME, self.CHANGE_TYPE_FIELD]
        for _trading_date in tqdm(sorted(history_companies[self.DATE_FIELD_NAME].unique(), reverse=True)):
            _currenet_code = history_companies[history_companies[self.DATE_FIELD_NAME] == _trading_date][
                self.SYMBOL_FIELD_NAME
            ].tolist()
            if last_code:
                add_code = list(set(last_code) - set(_currenet_code))
                remote_code = list(set(_currenet_code) - set(last_code))
                for _code in add_code:
                    result_df_list.append(
                        pd.DataFrame(
                            [[get_trading_date_by_shift(self.calendar_list, _trading_date, 1), _code, self.ADD]],
                            columns=_columns,
                        )
                    )
                for _code in remote_code:
                    result_df_list.append(
                        pd.DataFrame(
                            [[get_trading_date_by_shift(self.calendar_list, _trading_date, 0), _code, self.REMOVE]],
                            columns=_columns,
                        )
                    )
            last_code = _currenet_code
        df = pd.concat(result_df_list)
        logger.info("end of parse changes from history companies.")
        return df

    def parse_instruments(self):
        """parse instruments, eg: csi300.txt

        Examples
        -------
            $ python collector.py parse_instruments --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        logger.info(f"start parse {self.index_name.lower()} companies.....")
        instruments_columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]
        changers_df = self.get_changes()
        new_df = self.get_new_companies()
        if new_df is None or new_df.empty:
            raise ValueError(f"get new companies error: {self.index_name}")
        new_df = new_df.copy()
        logger.info("parse history companies by changes......")
        for _row in tqdm(changers_df.sort_values(self.DATE_FIELD_NAME, ascending=False).itertuples(index=False)):
            if _row.type == self.ADD:
                min_end_date = new_df.loc[new_df[self.SYMBOL_FIELD_NAME] == _row.symbol, self.END_DATE_FIELD].min()
                new_df.loc[
                    (new_df[self.END_DATE_FIELD] == min_end_date) & (new_df[self.SYMBOL_FIELD_NAME] == _row.symbol),
                    self.START_DATE_FIELD,
                ] = _row.date
            else:
                _tmp_df = pd.DataFrame([[_row.symbol, self.bench_start_date, _row.date]], columns=instruments_columns)
                new_df = pd.concat([new_df, _tmp_df], sort=False)

        inst_df = new_df.loc[:, instruments_columns]
        _inst_prefix = self.INST_PREFIX.strip()
        if _inst_prefix:
            inst_df["save_inst"] = inst_df[self.SYMBOL_FIELD_NAME].apply(lambda x: f"{_inst_prefix}{x}")
        inst_df = self.format_datetime(inst_df)
        inst_df.to_csv(
            self.instruments_dir.joinpath(f"{self.index_name.lower()}.txt"), sep="\t", index=False, header=None
        )
        logger.info(f"parse {self.index_name.lower()} companies finished.")
