import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional, Callable


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
    _count = int(match_obj.group(1) if match_obj.group(1) else "1")
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


def sample_calendar(calendar_raw: np.ndarray, freq_raw: str, freq_sam: str) -> np.ndarray:
    """
    Sample the calendar with frequency freq_raw into the calendar with frequency freq_sam

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
    if freq_sam == "minute":

        def cal_next_sam_minute(x, sam_minutes):
            hour = x.hour
            minute = x.minute
            if (hour == 9 and minute >= 30) or (9 < hour < 11) or (hour == 11 and minute < 30):
                minute_index = (hour - 9) * 60 + minute - 30
            elif 13 <= hour < 15:
                minute_index = (hour - 13) * 60 + minute + 120
            else:
                raise ValueError("calendar hour must be in [9, 11] or [13, 15]")

            minute_index = minute_index // sam_minutes * sam_minutes

            if 0 <= minute_index < 120:
                return 9 + (minute_index + 30) // 60, (minute_index + 30) % 60
            elif 120 <= minute_index < 240:
                return 13 + (minute_index - 120) // 60, (minute_index - 120) % 60
            else:
                raise ValueError("calendar minute_index error")

        if freq_raw != "minute":
            raise ValueError("when sampling minute calendar, freq of raw calendar must be minute or min")
        else:
            if raw_count > sam_count:
                raise ValueError("raw freq must be higher than sampling freq")
        _calendar_minute = np.unique(
            list(
                map(lambda x: pd.Timestamp(x.year, x.month, x.day, *cal_next_sam_minute(x, sam_count), 0), calendar_raw)
            )
        )
        if calendar_raw[0] > _calendar_minute[0]:
            _calendar_minute[0] = calendar_raw[0]
        return _calendar_minute
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


def get_sample_freq_calendar(
    start_time: Union[str, pd.Timestamp] = None,
    end_time: Union[str, pd.Timestamp] = None,
    freq: str = "day",
    future: bool = False,
) -> Tuple[np.ndarray, str, Optional[str]]:
    """
    Get the calendar with frequency freq.

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
                    start_time=start_time, end_time=end_time, freq="min", freq_sam=freq, future=future
                )
                freq = "min"
        elif norm_freq == "minute":
            _calendar = Cal.calendar(start_time=start_time, end_time=end_time, freq="min", freq_sam=freq, future=future)
            freq = "min"
        else:
            raise ValueError(f"freq {freq} is not supported")
    return _calendar, freq, freq_sam


def sample_feature(
    feature: Union[pd.DataFrame, pd.Series],
    start_time: Union[str, pd.Timestamp] = None,
    end_time: Union[str, pd.Timestamp] = None,
    fields: Union[str, List[str]] = None,
    method: Union[str, Callable] = "last",
    method_kwargs: dict = {},
):
    """
    Sample value from pandas DataFrame or Series for each stock

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

                print(sample_feature(feature, start_time="2010-01-04", end_time="2010-01-05", fields=["$close", "$volume"], method="last"))
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

                print(sample_feature(feature, start_time="2010-01-04", end_time="2010-01-05", fields=["$close", "$volume"], method="last"))

                $close 87.433578
                $volume 28117442.0

                print(sample_feature(feature, start_time="2010-01-04", end_time="2010-01-05", fields="$close", method="last"))

                87.433578

    Parameters
    ----------
    feature : Union[pd.DataFrame, pd.Series]
        Raw feature to be sampled
    start_time : Union[str, pd.Timestamp], optional
        start sampling time, by default None
    end_time : Union[str, pd.Timestamp], optional
        end sampling time, by default None
    fields : Union[str, List[str]], optional
        column names, it's ignored when sample pd.Series data, by default None(all columns)
    method : Union[str, Callable], optional
        sample method, apply method function to each stock series data, by default "last"
        - If type(method) is str, it should be an attribute of SeriesGroupBy or DataFrameGroupby, and run feature.groupby
        - If `feature` has MultiIndex[instrument, datetime], method must be a member of pandas.groupby when it's type is str.or callable function.
    method_kwargs : dict, optional
        arguments of method, by default {}

    Returns
    -------
        The Sampled DataFrame/Series/Value
    """

    selector_datetime = slice(start_time, end_time)
    if fields is None:
        fields = slice(None)

    from ..data.dataset.utils import get_level_index

    datetime_level = get_level_index(feature, level="datetime") == 0
    if isinstance(feature, pd.Series):
        feature = feature.loc[selector_datetime] if datetime_level else feature.loc[(slice(None), selector_datetime)]
    elif isinstance(feature, pd.DataFrame):
        feature = (
            feature.loc[selector_datetime, fields]
            if datetime_level
            else feature.loc[(slice(None), selector_datetime), fields]
        )
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
