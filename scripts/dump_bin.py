# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import shutil
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger


class DumpData(object):
    FILE_SUFFIX = ".csv"

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        works: int = None,
        date_field_name: str = "date",
    ):
        """

        Parameters
        ----------
        csv_path: str
            stock data path or directory
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        works: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        """
        csv_path = Path(csv_path).expanduser()
        self.csv_files = sorted(csv_path.glob(f"*{self.FILE_SUFFIX}") if csv_path.is_dir() else [csv_path])
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        self.freq = freq
        self.calendar_format = "%Y-%m-%d" if self.freq == "day" else "%Y-%m-%d %H:%M:%S"

        self.works = works
        self.date_field_name = date_field_name

        self._calendars_dir = self.qlib_dir.joinpath("calendars")
        self._features_dir = self.qlib_dir.joinpath("features")
        self._instruments_dir = self.qlib_dir.joinpath("instruments")

        self._calendars_list = []
        self._include_fields = ()
        self._exclude_fields = ()

    def _backup_qlib_dir(self, target_dir: Path):
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def _get_date_for_df(self, file_path: Path, *, is_begin_end: bool = False):
        df = pd.read_csv(str(file_path.resolve()))
        if df.empty or self.date_field_name not in df.columns.tolist():
            return []
        if is_begin_end:
            return [df[self.date_field_name].min(), df[self.date_field_name].max()]
        return df[self.date_field_name].tolist()

    def _get_source_data(self, file_path: Path):
        df = pd.read_csv(str(file_path.resolve()))
        df[self.date_field_name] = df[self.date_field_name].astype(np.datetime64)
        return df

    def _file_to_bin(self, file_path: Path = None):
        code = file_path.name[: -len(self.FILE_SUFFIX)].strip().lower()
        features_dir = self._features_dir.joinpath(code)
        features_dir.mkdir(parents=True, exist_ok=True)
        calendars_df = pd.DataFrame(data=self._calendars_list, columns=[self.date_field_name])
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype(np.datetime64)
        # read csv file
        df = self._get_source_data(file_path)
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        r_df = df.reindex(cal_df.index)
        date_index = self._calendars_list.index(r_df.index.min())
        for field in (
            self._include_fields
            if self._include_fields
            else set(r_df.columns) - set(self._exclude_fields)
            if self._exclude_fields
            else r_df.columns
        ):

            bin_path = features_dir.joinpath(f"{field}.{self.freq}.bin")
            if field not in r_df.columns:
                continue
            r = np.hstack([date_index, r_df[field]]).astype("<f")
            r.tofile(str(bin_path.resolve()))

    @staticmethod
    def _read_calendar(calendar_path: Path):
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def dump_features(
        self,
        calendar_path: str = None,
        include_fields: tuple = None,
        exclude_fields: tuple = None,
    ):
        """dump features

        Parameters
        ---------
        calendar_path: str
            calendar path

        include_fields: str
            dump fields

        exclude_fields: str
            fields not dumped

        Notes
        ---------
        python dump_bin.py dump_features --csv_path <stock data directory or path> --qlib_dir <dump data directory>

        Examples
        ---------

        # dump all stock
        python dump_bin.py dump_features --csv_path ~/tmp/stock_data --qlib_dir ~/tmp/qlib_data --exclude_fields date,code,timestamp,code_name
        # dump one stock
        python dump_bin.py dump_features --csv_path ~/tmp/stock_data/sh600000.csv --qlib_dir ~/tmp/qlib_data --calendar_path ~/tmp/qlib_data/calendar/all.txt --exclude_fields date,code,timestamp,code_name
        """
        logger.info("start dump features......")
        if calendar_path is not None:
            # read calendar from calendar file
            self._calendars_list = self._read_calendar(Path(calendar_path))

        if not self._calendars_list:
            self.dump_calendars()

        self._include_fields = tuple(map(str.strip, include_fields)) if include_fields else self._include_fields
        self._exclude_fields = tuple(map(str.strip, exclude_fields)) if exclude_fields else self._exclude_fields
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(self._file_to_bin, self.csv_files):
                    p_bar.update()

        logger.info("end of features dump.\n")

    def dump_calendars(self):
        """dump calendars

        Notes
        ---------
        python dump_bin.py dump_calendars --csv_path <stock data directory or path> --qlib_dir <dump data directory>

        Examples
        ---------
        python dump_bin.py dump_calendars --csv_path ~/tmp/stock_data --qlib_dir ~/tmp/qlib_data
        """
        logger.info("start dump calendars......")
        calendar_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        all_datetime = set()
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for temp_datetime in executor.map(self._get_date_for_df, self.csv_files):
                    all_datetime = all_datetime | set(temp_datetime)
                    p_bar.update()

        self._calendars_list = sorted(map(pd.Timestamp, all_datetime))
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        result_calendar_list = list(map(lambda x: x.strftime(self.calendar_format), self._calendars_list))
        np.savetxt(calendar_path, result_calendar_list, fmt="%s", encoding="utf-8")
        logger.info("end of calendars dump.\n")

    def dump_instruments(self):
        """dump instruments

        Notes
        ---------
        python dump_bin.py dump_instruments --csv_path <stock data directory or path> --qlib_dir <dump data directory>

        Examples
        ---------
        python dump_bin.py dump_instruments --csv_path ~/tmp/stock_data --qlib_dir ~/tmp/qlib_data
        """
        logger.info("start dump instruments......")
        symbol_list = list(map(lambda x: x.name[: -len(self.FILE_SUFFIX)], self.csv_files))
        _result_list = []
        _fun = partial(self._get_date_for_df, is_begin_end=True)
        with tqdm(total=len(symbol_list)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as execute:
                for symbol, res in zip(symbol_list, execute.map(_fun, self.csv_files)):
                    if res:
                        begin_time = res[0]
                        end_time = res[-1]
                        _result_list.append(f"{symbol.upper()}\t{begin_time}\t{end_time}")
                    p_bar.update()

        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        to_path = str(self._instruments_dir.joinpath("all.txt").resolve())
        np.savetxt(to_path, _result_list, fmt="%s", encoding="utf-8")
        logger.info("end of instruments dump.\n")

    def dump(self, include_fields: str = None, exclude_fields: tuple = None):
        """dump data

        Parameters
        ----------
        include_fields: str
            dump fields

        exclude_fields: str
            fields not dumped

        Examples
        ---------
        python dump_bin.py dump --csv_path ~/tmp/stock_data --qlib_dir ~/tmp/qlib_data --include_fields open,close,high,low,volume,factor
        python dump_bin.py dump --csv_path ~/tmp/stock_data --qlib_dir ~/tmp/qlib_data --exclude_fields date,code,timestamp,code_name
        """
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self.dump_calendars()
        self.dump_features(include_fields=include_fields, exclude_fields=exclude_fields)
        self.dump_instruments()


if __name__ == "__main__":
    fire.Fire(DumpData)
