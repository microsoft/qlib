import numpy as np
import pandas as pd

from functools import partial
from typing import Union, Callable

from . import lazy_sort_index
from .time import Freq, cal_sam_minute


def resam_calendar(calendar_raw: np.ndarray, freq_raw: Union[str, Freq], freq_sam: Union[str, Freq]) -> np.ndarray:
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
    freq_raw = Freq(freq_raw)
    freq_sam = Freq(freq_sam)
    if not len(calendar_raw):
        return calendar_raw

    # if freq_sam is xminute, divide each trading day into several bars evenly
    if freq_sam.base == Freq.NORM_FREQ_MINUTE:
        if freq_raw.base != Freq.NORM_FREQ_MINUTE:
            raise ValueError("when sampling minute calendar, freq of raw calendar must be minute or min")
        else:
            if freq_raw.count > freq_sam.count:
                raise ValueError("raw freq must be higher than sampling freq")
        _calendar_minute = np.unique(list(map(lambda x: cal_sam_minute(x, freq_sam.count), calendar_raw)))
        return _calendar_minute

    # else, convert the raw calendar into day calendar, and divide the whole calendar into several bars evenly
    else:
        _calendar_day = np.unique(list(map(lambda x: pd.Timestamp(x.year, x.month, x.day, 0, 0, 0), calendar_raw)))
        if freq_sam.base == Freq.NORM_FREQ_DAY:
            return _calendar_day[:: freq_sam.count]

        elif freq_sam.base == Freq.NORM_FREQ_WEEK:
            _day_in_week = np.array(list(map(lambda x: x.dayofweek, _calendar_day)))
            _calendar_week = _calendar_day[np.ediff1d(_day_in_week, to_begin=-1) < 0]
            return _calendar_week[:: freq_sam.count]

        elif freq_sam.base == Freq.NORM_FREQ_MONTH:
            _day_in_month = np.array(list(map(lambda x: x.day, _calendar_day)))
            _calendar_month = _calendar_day[np.ediff1d(_day_in_month, to_begin=-1) < 0]
            return _calendar_month[:: freq_sam.count]
        else:
            raise ValueError("sampling freq must be xmin, xd, xw, xm")


def get_higher_eq_freq_feature(instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1):
    """get the feature with higher or equal frequency than `freq`.
    Returns
    -------
    pd.DataFrame
        the feature with higher or equal frequency
    """

    from ..data.data import D

    try:
        _result = D.features(instruments, fields, start_time, end_time, freq=freq, disk_cache=disk_cache)
        _freq = freq
    except (ValueError, KeyError):
        _, norm_freq = Freq.parse(freq)
        if norm_freq in [Freq.NORM_FREQ_MONTH, Freq.NORM_FREQ_WEEK, Freq.NORM_FREQ_DAY]:
            try:
                _result = D.features(instruments, fields, start_time, end_time, freq="day", disk_cache=disk_cache)
                _freq = "day"
            except (ValueError, KeyError):
                _result = D.features(instruments, fields, start_time, end_time, freq="1min", disk_cache=disk_cache)
                _freq = "1min"
        elif norm_freq == Freq.NORM_FREQ_MINUTE:
            _result = D.features(instruments, fields, start_time, end_time, freq="1min", disk_cache=disk_cache)
            _freq = "1min"
        else:
            raise ValueError(f"freq {freq} is not supported")
    return _result, _freq


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
    ts_feature : Union[pd.DataFrame, pd.Series]
        Raw time-series feature to be resampled
    start_time : Union[str, pd.Timestamp], optional
        start sampling time, by default None
    end_time : Union[str, pd.Timestamp], optional
        end sampling time, by default None
    method : Union[str, Callable], optional
        sample method, apply method function to each stock series data, by default "last"
        - If type(method) is str or callable function, it should be an attribute of SeriesGroupBy or DataFrameGroupby, and applies groupy.method for the sliced time-series data
        - If method is None, do nothing for the sliced time-series data.
    method_kwargs : dict, optional
        arguments of method, by default {}

    Returns
    -------
        The resampled DataFrame/Series/value, return None when the resampled data is empty.
    """

    selector_datetime = slice(start_time, end_time)

    from ..data.dataset.utils import get_level_index

    feature = lazy_sort_index(ts_feature)

    datetime_level = get_level_index(feature, level="datetime") == 0
    if datetime_level:
        feature = feature.loc[selector_datetime]
    else:
        feature = feature.loc(axis=0)[(slice(None), selector_datetime)]

    if feature.empty:
        return None
    if isinstance(feature.index, pd.MultiIndex):
        if callable(method):
            method_func = method
            return feature.groupby(level="instrument").apply(method_func, **method_kwargs)
        elif isinstance(method, str):
            return getattr(feature.groupby(level="instrument"), method)(**method_kwargs)
    else:
        if callable(method):
            method_func = method
            return method_func(feature, **method_kwargs)
        elif isinstance(method, str):
            return getattr(feature, method)(**method_kwargs)
    return feature


def get_valid_value(series, last=True):
    """get the first/last not nan value of pd.Series with single level index
    Parameters
    ----------
    series : pd.Series
        series should not be empty
    last : bool, optional
        whether to get the last valid value, by default True
        - if last is True, get the last valid value
        - else, get the first valid value

    Returns
    -------
    Nan | float
        the first/last valid value
    """
    return series.fillna(method="ffill").iloc[-1] if last else series.fillna(method="bfill").iloc[0]


def _ts_data_valid(ts_feature, last=False):
    """get the first/last not nan value of pd.Series|DataFrame with single level index"""
    if isinstance(ts_feature, pd.DataFrame):
        return ts_feature.apply(lambda column: get_valid_value(column, last=last))
    elif isinstance(ts_feature, pd.Series):
        return get_valid_value(ts_feature, last=last)
    else:
        raise TypeError(f"ts_feature should be pd.DataFrame/Series, not {type(ts_feature)}")


ts_data_last = partial(_ts_data_valid, last=True)
ts_data_first = partial(_ts_data_valid, last=False)
