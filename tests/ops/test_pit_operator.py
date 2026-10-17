# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import pytest

from qlib.data.pit import P, Cal


class _PeriodFeature:
    def get_extended_window_size(self):
        return 0, 0


class _ValueP(P):
    def __init__(self, positions):
        super().__init__(_PeriodFeature())
        self.positions = positions

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return pd.Series([float(self.positions[pd.Timestamp(cur_time)])])


class _MissingP(P):
    def __init__(self):
        super().__init__(_PeriodFeature())

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        raise FileNotFoundError(instrument)


class _FutureFeature:
    def get_extended_window_size(self):
        return 0, 1


def _positions(calendar):
    return {pd.Timestamp(ts): i for i, ts in enumerate(calendar)}


def _patch_calendar(monkeypatch, calendar):
    def calendar_fn(start_time=None, end_time=None, freq="day", future=False):
        return calendar

    monkeypatch.setattr(Cal, "calendar", calendar_fn, raising=False)


def _patch_locate(monkeypatch, calendar):
    positions = _positions(calendar)

    def locate_index(start_time, end_time, freq="day", future=False):
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        return start_time, end_time, positions[start_time], positions[end_time]

    monkeypatch.setattr(Cal, "locate_index", locate_index, raising=False)


def test_integer_bounds_keep_a_range_index(monkeypatch):
    calendar = pd.bdate_range("2020-01-01", periods=5)
    _patch_calendar(monkeypatch, calendar)
    series = _ValueP(_positions(calendar))._load_internal("SH600000", 1, 3, "day")
    assert isinstance(series.index, pd.RangeIndex)
    assert list(series.index) == [1, 2, 3]
    assert series.tolist() == [1.0, 2.0, 3.0]


def test_datetime_bounds_align_and_keep_the_datetime_index(monkeypatch):
    calendar = pd.bdate_range("2020-01-01", periods=5)
    _patch_calendar(monkeypatch, calendar)
    _patch_locate(monkeypatch, calendar)
    start, end = calendar[1], calendar[3]
    series = _ValueP(_positions(calendar))._load_internal("SH600000", start, end, "day")
    selected = series.loc[start:end]
    assert list(selected.index) == list(calendar[1:4])
    assert selected.tolist() == [1.0, 2.0, 3.0]


def test_missing_period_data_returns_an_empty_series(monkeypatch):
    calendar = pd.bdate_range("2020-01-01", periods=4)
    _patch_calendar(monkeypatch, calendar)
    series = _MissingP()._load_internal("SH600000", 0, 2, "day")
    assert series.empty
    assert series.dtype == "float32"


def test_future_period_reference_is_rejected(monkeypatch):
    calendar = pd.bdate_range("2020-01-01", periods=4)
    _patch_calendar(monkeypatch, calendar)
    operator = P(_FutureFeature())
    with pytest.raises(ValueError, match="does not support referring to future period"):
        operator._load_internal("SH600000", 0, 1, "day")
