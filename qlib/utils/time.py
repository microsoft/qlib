# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Time related utils are compiled in this script
"""
import bisect
from datetime import datetime, time, date
from typing import List, Optional, Tuple, Union
import functools
import re

import pandas as pd

from qlib.config import C


@functools.lru_cache(maxsize=240)
def get_min_cal(shift: int = 0) -> List[time]:
    """
    get the minute level calendar in day period

    Parameters
    ----------
    shift : int
        the shift direction would be like pandas shift.
        series.shift(1) will replace the value at `i`-th with the one at `i-1`-th

    Returns
    -------
    List[time]:

    """
    cal = []
    for ts in list(pd.date_range("9:30", "11:29", freq="1min") - pd.Timedelta(minutes=shift)) + list(
        pd.date_range("13:00", "14:59", freq="1min") - pd.Timedelta(minutes=shift)
    ):
        cal.append(ts.time())
    return cal


def is_single_value(start_time, end_time, freq, region="cn"):
    """Is there only one piece of data for stock market.

    Parameters
    ----------
    start_time : Union[pd.Timestamp, str]
        closed start time for data.
    end_time : Union[pd.Timestamp, str]
        closed end time for data.
    freq :
    Returns
    -------
    bool
        True means one piece of data to obtain.
    """
    if region == "cn":
        if end_time - start_time < freq:
            return True
        if start_time.hour == 11 and start_time.minute == 29 and start_time.second == 0:
            return True
        if start_time.hour == 14 and start_time.minute == 59 and start_time.second == 0:
            return True
        return False
    else:
        raise NotImplementedError(f"please implement the is_single_value func for {region}")


class Freq:
    NORM_FREQ_MONTH = "month"
    NORM_FREQ_WEEK = "week"
    NORM_FREQ_DAY = "day"
    NORM_FREQ_MINUTE = "min"  # using min instead of minute for align with Qlib's data filename
    SUPPORT_CAL_LIST = [NORM_FREQ_MINUTE, NORM_FREQ_DAY]  # FIXME: this list should from data

    MIN_CAL = get_min_cal()

    def __init__(self, freq: Union[str, "Freq"]) -> None:
        if isinstance(freq, str):
            self.count, self.base = self.parse(freq)
        elif isinstance(freq, Freq):
            self.count, self.base = freq.count, freq.base
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def __eq__(self, freq):
        freq = Freq(freq)
        return freq.count == self.count and freq.base == self.base

    def __str__(self):
        # trying to align to the filename of Qlib: day, 30min, 5min, 1min...
        return f"{self.count if self.count != 1 or self.base != 'day' else ''}{self.base}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    @staticmethod
    def parse(freq: str) -> Tuple[int, str]:
        """
        Parse freq into a unified format

        Parameters
        ----------
        freq : str
            Raw freq, supported freq should match the re '^([0-9]*)(month|mon|week|w|day|d|minute|min)$'

        Returns
        -------
        freq: Tuple[int, str]
            Unified freq, including freq count and unified freq unit. The freq unit should be '[month|week|day|minute]'.
                Example:

                .. code-block::

                    print(Freq.parse("day"))
                    (1, "day" )
                    print(Freq.parse("2mon"))
                    (2, "month")
                    print(Freq.parse("10w"))
                    (10, "week")

        """
        freq = freq.lower()
        match_obj = re.match("^([0-9]*)(month|mon|week|w|day|d|minute|min)$", freq)
        if match_obj is None:
            raise ValueError(
                "freq format is not supported, the freq should be like (n)month/mon, (n)week/w, (n)day/d, (n)minute/min"
            )
        _count = int(match_obj.group(1)) if match_obj.group(1) else 1
        _freq = match_obj.group(2)
        _freq_format_dict = {
            "month": Freq.NORM_FREQ_MONTH,
            "mon": Freq.NORM_FREQ_MONTH,
            "week": Freq.NORM_FREQ_WEEK,
            "w": Freq.NORM_FREQ_WEEK,
            "day": Freq.NORM_FREQ_DAY,
            "d": Freq.NORM_FREQ_DAY,
            "minute": Freq.NORM_FREQ_MINUTE,
            "min": Freq.NORM_FREQ_MINUTE,
        }
        return _count, _freq_format_dict[_freq]

    @staticmethod
    def get_timedelta(n: int, freq: str) -> pd.Timedelta:
        """
        get pd.Timedeta object

        Parameters
        ----------
        n : int
        freq : str
            Typically, they are the return value of Freq.parse

        Returns
        -------
        pd.Timedelta:
        """
        return pd.Timedelta(f"{n}{freq}")

    @staticmethod
    def get_min_delta(left_frq: str, right_freq: str):
        """Calculate freq delta

        Parameters
        ----------
        left_frq: str
        right_freq: str

        Returns
        -------

        """
        minutes_map = {
            Freq.NORM_FREQ_MINUTE: 1,
            Freq.NORM_FREQ_DAY: 60 * 24,
            Freq.NORM_FREQ_WEEK: 7 * 60 * 24,
            Freq.NORM_FREQ_MONTH: 30 * 7 * 60 * 24,
        }
        left_freq = Freq(left_frq)
        left_minutes = left_freq.count * minutes_map[left_freq.base]
        right_freq = Freq(right_freq)
        right_minutes = right_freq.count * minutes_map[right_freq.base]
        return left_minutes - right_minutes

    @staticmethod
    def get_recent_freq(base_freq: Union[str, "Freq"], freq_list: List[Union[str, "Freq"]]) -> Optional["Freq"]:
        """Get the closest freq to base_freq from freq_list

        Parameters
        ----------
        base_freq
        freq_list

        Returns
        -------
        if the recent frequency is found
            Freq
        else:
            None
        """
        base_freq = Freq(base_freq)
        # use the nearest freq greater than 0
        _freq_minutes = []
        min_freq = None
        for _freq in freq_list:
            freq = Freq(_freq)
            _min_delta = Freq.get_min_delta(base_freq, _freq)
            if _min_delta < 0:
                continue
            if min_freq is None:
                min_freq = (_min_delta, str(_freq))
                continue
            min_freq = min_freq if min_freq[0] <= _min_delta else (_min_delta, _freq)
        return min_freq[1] if min_freq else None


CN_TIME = [
    datetime.strptime("9:30", "%H:%M"),
    datetime.strptime("11:30", "%H:%M"),
    datetime.strptime("13:00", "%H:%M"),
    datetime.strptime("15:00", "%H:%M"),
]
US_TIME = [datetime.strptime("9:30", "%H:%M"), datetime.strptime("16:00", "%H:%M")]


def time_to_day_index(time_obj: Union[str, datetime], region: str = "cn"):
    if isinstance(time_obj, str):
        time_obj = datetime.strptime(time_obj, "%H:%M")

    if region == "cn":
        if CN_TIME[0] <= time_obj < CN_TIME[1]:
            return int((time_obj - CN_TIME[0]).total_seconds() / 60)
        elif CN_TIME[2] <= time_obj < CN_TIME[3]:
            return int((time_obj - CN_TIME[2]).total_seconds() / 60) + 120
        else:
            raise ValueError(f"{time_obj} is not the opening time of the {region} stock market")
    elif region == "us":
        if US_TIME[0] <= time_obj < US_TIME[1]:
            return int((time_obj - US_TIME[0]).total_seconds() / 60)
        else:
            raise ValueError(f"{time_obj} is not the opening time of the {region} stock market")
    else:
        raise ValueError(f"{region} is not supported")


def get_day_min_idx_range(start: str, end: str, freq: str) -> Tuple[int, int]:
    """
    get the min-bar index in a day for a time range (both left and right is closed) given a fixed frequency
    Parameters
    ----------
    start : str
        e.g. "9:30"
    end : str
        e.g. "14:30"
    freq : str
        "1min"

    Returns
    -------
    Tuple[int, int]:
        The index of start and end in the calendar. Both left and right are **closed**
    """
    start = pd.Timestamp(start).time()
    end = pd.Timestamp(end).time()
    freq = Freq(freq)
    in_day_cal = Freq.MIN_CAL[:: freq.count]
    left_idx = bisect.bisect_left(in_day_cal, start)
    right_idx = bisect.bisect_right(in_day_cal, end) - 1
    return left_idx, right_idx


def concat_date_time(date_obj: date, time_obj: time) -> pd.Timestamp:
    return pd.Timestamp(
        datetime(
            date_obj.year,
            month=date_obj.month,
            day=date_obj.day,
            hour=time_obj.hour,
            minute=time_obj.minute,
            second=time_obj.second,
            microsecond=time_obj.microsecond,
        )
    )


def cal_sam_minute(x: pd.Timestamp, sam_minutes: int) -> pd.Timestamp:
    """
    align the minute-level data to a down sampled calendar

    e.g. align 10:38 to 10:35 in 5 minute-level(10:30 in 10 minute-level)

    Parameters
    ----------
    x : pd.Timestamp
        datetime to be aligned
    sam_minutes : int
        align to `sam_minutes` minute-level calendar

    Returns
    -------
    pd.Timestamp:
        the datetime after aligned
    """
    cal = get_min_cal(C.min_data_shift)[::sam_minutes]
    idx = bisect.bisect_right(cal, x.time()) - 1
    _date, new_time = x.date(), cal[idx]
    return concat_date_time(_date, new_time)


def epsilon_change(date_time: pd.Timestamp, direction: str = "backward") -> pd.Timestamp:
    """
    change the time by infinitely small quantity.


    Parameters
    ----------
    date_time : pd.Timestamp
        the original time
    direction : str
        the direction the time are going to
        - "backward" for going to history
        - "forward" for going to the future

    Returns
    -------
    pd.Timestamp:
        the shifted time
    """
    if direction == "backward":
        return date_time - pd.Timedelta(seconds=1)
    elif direction == "forward":
        return date_time + pd.Timedelta(seconds=1)
    else:
        raise ValueError("Wrong input")


if __name__ == "__main__":
    print(get_day_min_idx_range("8:30", "14:59", "10min"))
