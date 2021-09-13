# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from typing import List
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger

# get data from baostock
import baostock as bs

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent.parent))


from data_collector.utils import generate_minutes_calendar_from_daily


def read_calendar_from_qlib(qlib_dir: Path) -> pd.DataFrame:
    calendar_path = qlib_dir.joinpath("calendars").joinpath("day.txt")
    if not calendar_path.exists():
        return pd.DataFrame()
    return pd.read_csv(calendar_path, header=None)


def write_calendar_to_qlib(qlib_dir: Path, date_list: List[str], freq: str = "day"):
    calendar_path = str(qlib_dir.joinpath("calendars").joinpath(f"{freq}_future.txt"))

    np.savetxt(calendar_path, date_list, fmt="%s", encoding="utf-8")
    logger.info(f"write future calendars success: {calendar_path}")


def generate_qlib_calendar(date_list: List[str], freq: str) -> List[str]:
    print(freq)
    if freq == "day":
        return date_list
    elif freq == "1min":
        date_list = generate_minutes_calendar_from_daily(date_list, freq=freq).tolist()
        return list(map(lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S"), date_list))
    else:
        raise ValueError(f"Unsupported freq: {freq}")


def future_calendar_collector(qlib_dir: [str, Path], freq: str = "day"):
    """get future calendar

    Parameters
    ----------
    qlib_dir: str or Path
        qlib data directory
    freq: str
        value from ["day", "1min"], by default day
    """
    qlib_dir = Path(qlib_dir).expanduser().resolve()
    if not qlib_dir.exists():
        raise FileNotFoundError(str(qlib_dir))

    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"login error: {lg.error_msg}")
        return
    # read daily calendar
    daily_calendar = read_calendar_from_qlib(qlib_dir)
    end_year = pd.Timestamp.now().year
    if daily_calendar.empty:
        start_year = pd.Timestamp.now().year
    else:
        start_year = pd.Timestamp(daily_calendar.iloc[-1, 0]).year
    rs = bs.query_trade_dates(start_date=pd.Timestamp(f"{start_year}-01-01"), end_date=f"{end_year}-12-31")
    data_list = []
    while (rs.error_code == "0") & rs.next():
        _row_data = rs.get_row_data()
        if int(_row_data[1]) == 1:
            data_list.append(_row_data[0])
    data_list = sorted(data_list)
    date_list = generate_qlib_calendar(data_list, freq=freq)
    date_list = sorted(set(daily_calendar.loc[:, 0].values.tolist() + date_list))
    write_calendar_to_qlib(qlib_dir, date_list, freq=freq)
    bs.logout()
    logger.info(f"get trading dates success: {start_year}-01-01 to {end_year}-12-31")


if __name__ == "__main__":
    fire.Fire(future_calendar_collector)
