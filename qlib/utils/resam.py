import re
import datetime

import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional, Callable

from . import lazy_sort_index
from ..config import C


def parse_freq(freq: str) -> Tuple[int, str]:
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

                print(parse_freq("day"))
                (1, "day" )
                print(parse_freq("2mon"))
                (2, "month")
                print(parse_freq("10w"))
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
        "month": "month",
        "mon": "month",
        "week": "week",
        "w": "week",
        "day": "day",
        "d": "day",
        "minute": "minute",
        "min": "minute",
    }
    return _count, _freq_format_dict[_freq]


def resam_calendar(calendar_raw: np.ndarray, freq_raw: str, freq_sam: str) -> np.ndarray:
    """
    Resample the calendar with frequency freq_raw into the calendar with frequency freq_sam
    Assumption:
        - Fix length (240) of the calendar in each day.

    Parameters
    ----------
    calendar_raw : np.ndarray
        The calendar with frequency  freq_raw
    freq_raw : str
        Frequency of the raw calendar
    freq_sam : str
        Sample frequency

    Returns
    -------
    np.ndarray
        The calendar with frequency freq_sam
    """
    raw_count, freq_raw = parse_freq(freq_raw)
    sam_count, freq_sam = parse_freq(freq_sam)
    if not len(calendar_raw):
        return calendar_raw

    # if freq_sam is xminute, divide each trading day into several bars evenly
    if freq_sam == "minute":

        def cal_sam_minute(x, sam_minutes):
            """
            Sample raw calendar into calendar with sam_minutes freq, shift represents the shift minute the market time
                - open time of stock market is [9:30 - shift*pd.Timedelta(minutes=1)]
                - mid close time of stock market is [11:29 - shift*pd.Timedelta(minutes=1)]
                - mid open time of stock market is [13:00 - shift*pd.Timedelta(minutes=1)]
                - close time of stock market is [14:59 - shift*pd.Timedelta(minutes=1)]
            """
            day_time = pd.Timestamp(x.date())
            shift = C.min_data_shift

            open_time = day_time + pd.Timedelta(hours=9, minutes=30) - shift * pd.Timedelta(minutes=1)
            mid_close_time = day_time + pd.Timedelta(hours=11, minutes=29) - shift * pd.Timedelta(minutes=1)
            mid_open_time = day_time + pd.Timedelta(hours=13, minutes=00) - shift * pd.Timedelta(minutes=1)
            close_time = day_time + pd.Timedelta(hours=14, minutes=59) - shift * pd.Timedelta(minutes=1)

            if open_time <= x <= mid_close_time:
                minute_index = (x - open_time).seconds // 60
            elif mid_open_time <= x <= close_time:
                minute_index = (x - mid_open_time).seconds // 60 + 120
            else:
                raise ValueError("datetime of calendar is out of range")
            minute_index = minute_index // sam_minutes * sam_minutes

            if 0 <= minute_index < 120:
                return open_time + minute_index * pd.Timedelta(minutes=1)
            elif 120 <= minute_index < 240:
                return mid_open_time + (minute_index - 120) * pd.Timedelta(minutes=1)
            else:
                raise ValueError("calendar minute_index error, check `min_data_shift` in qlib.config.C")

        if freq_raw != "minute":
            raise ValueError("when sampling minute calendar, freq of raw calendar must be minute or min")
        else:
            if raw_count > sam_count:
                raise ValueError("raw freq must be higher than sampling freq")
        _calendar_minute = np.unique(list(map(lambda x: cal_sam_minute(x, sam_count), calendar_raw)))
        return _calendar_minute

    # else, convert the raw calendar into day calendar, and divide the whole calendar into several bars evenly
    else:
        _calendar_day = np.unique(list(map(lambda x: pd.Timestamp(x.year, x.month, x.day, 0, 0, 0), calendar_raw)))
        if freq_sam == "day":
            return _calendar_day[::sam_count]

        elif freq_sam == "week":
            _day_in_week = np.array(list(map(lambda x: x.dayofweek, _calendar_day)))
            _calendar_week = _calendar_day[np.ediff1d(_day_in_week, to_begin=-1) < 0]
            return _calendar_week[::sam_count]

        elif freq_sam == "month":
            _day_in_month = np.array(list(map(lambda x: x.day, _calendar_day)))
            _calendar_month = _calendar_day[np.ediff1d(_day_in_month, to_begin=-1) < 0]
            return _calendar_month[::sam_count]
        else:
            raise ValueError("sampling freq must be xmin, xd, xw, xm")


def get_resam_calendar(
    start_time: Union[str, pd.Timestamp] = None,
    end_time: Union[str, pd.Timestamp] = None,
    freq: str = "day",
    future: bool = False,
) -> Tuple[np.ndarray, str, Optional[str]]:
    """
    Get the resampled calendar with frequency freq.

        - If the calendar with the raw frequency freq exists, return it directly

        - Else, sample from a higher frequency calendar automatically

    Parameters
    ----------
    start_time : Union[str, pd.Timestamp], optional
        start time of calendar, by default None
    end_time : Union[str, pd.Timestamp], optional
        end time of calendar, by default None
    freq : str, optional
        freq of calendar, by default "day"
    future : bool, optional
        whether including future trading day.

    Returns
    -------
    Tuple[np.ndarray, str, Optional[str]]

        - the first value is the calendar
        - the second value is the raw freq of calendar
        - the third value is the sampling freq of calendar, it's None if the raw frequency freq exists.

    """

    _, norm_freq = parse_freq(freq)

    from ..data.data import Cal

    try:
        _calendar = Cal.calendar(start_time=start_time, end_time=end_time, freq=freq, future=future)
        freq, freq_sam = freq, None
    except ValueError:
        freq_sam = freq
        if norm_freq in ["month", "week", "day"]:
            try:
                _calendar = Cal.calendar(
                    start_time=start_time, end_time=end_time, freq="day", freq_sam=freq, future=future
                )
                freq = "day"
            except ValueError:
                _calendar = Cal.calendar(
                    start_time=start_time, end_time=end_time, freq="1min", freq_sam=freq, future=future
                )
                freq = "min"
        elif norm_freq == "minute":
            _calendar = Cal.calendar(
                start_time=start_time, end_time=end_time, freq="1min", freq_sam=freq, future=future
            )
            freq = "min"
        else:
            raise ValueError(f"freq {freq} is not supported")
    return _calendar, freq, freq_sam


def resam_ts_data(
    ts_feature: Union[pd.DataFrame, pd.Series],
    start_time: Union[str, pd.Timestamp] = None,
    end_time: Union[str, pd.Timestamp] = None,
    method: Union[str, Callable] = "last",
    method_kwargs: dict = {},
):
    """
    Resample value from time-series data

        - If `feature` has MultiIndex[instrument, datetime], apply the `method` to each instruemnt data with datetime in [start_time, end_time]
            Example:

            .. code-block::

                print(feature)
                                        $close      $volume
                instrument  datetime
                SH600000    2010-01-04  86.778313   16162960.0
                            2010-01-05  87.433578   28117442.0
                            2010-01-06  85.713585   23632884.0
                            2010-01-07  83.788803   20813402.0
                            2010-01-08  84.730675   16044853.0

                SH600655    2010-01-04  2699.567383  158193.328125
                            2010-01-08  2612.359619   77501.406250
                            2010-01-11  2712.982422  160852.390625
                            2010-01-12  2788.688232  164587.937500
                            2010-01-13  2790.604004  145460.453125

                print(resam_ts_data(feature, start_time="2010-01-04", end_time="2010-01-05", fields=["$close", "$volume"], method="last"))
                            $close      $volume
                instrument
                SH600000    87.433578 28117442.0
                SH600655    2699.567383  158193.328125

        - Else, the `feature` should have Index[datetime], just apply the `method` to `feature` directly
            Example:

            .. code-block::
                print(feature)
                            $close      $volume
                datetime
                2010-01-04  86.778313   16162960.0
                2010-01-05  87.433578   28117442.0
                2010-01-06  85.713585   23632884.0
                2010-01-07  83.788803   20813402.0
                2010-01-08  84.730675   16044853.0

                print(resam_ts_data(feature, start_time="2010-01-04", end_time="2010-01-05", method="last"))

                $close 87.433578
                $volume 28117442.0

                print(resam_ts_data(feature['$close'], start_time="2010-01-04", end_time="2010-01-05", method="last"))

                87.433578

    Parameters
    ----------
    feature : Union[pd.DataFrame, pd.Series]
        Raw time-series feature to be resampled
    start_time : Union[str, pd.Timestamp], optional
        start sampling time, by default None
    end_time : Union[str, pd.Timestamp], optional
        end sampling time, by default None
    method : Union[str, Callable], optional
        sample method, apply method function to each stock series data, by default "last"
        - If type(method) is str, it should be an attribute of SeriesGroupBy or DataFrameGroupby, and run feature.groupby
        - If `feature` has MultiIndex[instrument, datetime], method must be a member of pandas.groupby when it's type is str.or callable function.
    method_kwargs : dict, optional
        arguments of method, by default {}

    Returns
    -------
        The Resampled DataFrame/Series/Value
    """

    selector_datetime = slice(start_time, end_time)

    from ..data.dataset.utils import get_level_index

    feature = lazy_sort_index(ts_feature)

    datetime_level = get_level_index(feature, level="datetime") == 0
    if datetime_level:
        feature = feature.loc[selector_datetime]
    else:
        feature = feature.loc[(slice(None), selector_datetime)]

    if feature.empty:
        return None
    if isinstance(feature.index, pd.MultiIndex):
        if callable(method):
            method_func = method
            return feature.groupby(level="instrument").apply(lambda x: method_func(x, **method_kwargs))
        elif isinstance(method, str):
            return getattr(feature.groupby(level="instrument"), method)(**method_kwargs)
    else:
        if callable(method):
            method_func = method
            return method_func(feature, **method_kwargs)
        elif isinstance(method, str):
            return getattr(feature, method)(**method_kwargs)
    return feature
