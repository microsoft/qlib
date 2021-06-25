# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Time related utils are compiled in this script
"""
import bisect
from datetime import time
from typing import List, Tuple
import re
from numpy import append
import pandas as pd


def get_min_cal() -> List[time]:
    """
    get the minute level calendar in day period

    Returns
    -------
    List[time]:

    """
    cal = []
    for ts in list(pd.date_range("9:30", "11:29", freq="1min")) + list(pd.date_range("13:00", "14:59", freq="1min")):
        cal.append(ts.time())
    return cal


class Freq:
    NORM_FREQ_MONTH = "month"
    NORM_FREQ_WEEK = "week"
    NORM_FREQ_DAY = "day"
    NORM_FREQ_MINUTE = "minute"
    SUPPORT_CAL_LIST = [NORM_FREQ_MINUTE]

    MIN_CAL = get_min_cal()

    def __init__(self, freq: str) -> None:
        self.count, self.base = self.parse(freq)

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
    in_day_cal = Freq.MIN_CAL[::freq.count]
    left_idx = bisect.bisect_left(in_day_cal, start)
    right_idx = bisect.bisect_right(in_day_cal, end) - 1
    return left_idx, right_idx


if __name__ == "__main__":
    print(get_day_min_idx_range("8:30", "14:59", "10min"))
