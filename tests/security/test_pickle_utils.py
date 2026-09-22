import io
import os
import pickle
from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from qlib.utils.pickle_utils import RestrictedUnpickler, get_safe_classes, restricted_pickle_loads


_BUSINESS_OFFSETS = [
    pytest.param(
        pd.offsets.BusinessHour(n=2, start=["08:30", "13:00"], end=["11:30", "16:00"], offset=timedelta(minutes=15)),
        id="split-business-hours",
    ),
    pytest.param(
        pd.offsets.CustomBusinessDay(
            n=2,
            weekmask="Mon Tue Thu Fri",
            holidays=["2024-01-04", "2024-01-15"],
            offset=timedelta(hours=1, minutes=15),
        ),
        id="custom-business-days",
    ),
]


class _MaliciousPayload:
    def __reduce__(self):
        return os.system, ("echo vulnerable",)


def _assert_rejects_without_execution(value, protocol, monkeypatch):
    payload = pickle.dumps(value, protocol=protocol)
    execute = Mock()
    module = os.system.__module__
    monkeypatch.setattr(f"{module}.system", execute)
    with pytest.raises(pickle.UnpicklingError, match=rf"Forbidden class: {module}\.system"):
        restricted_pickle_loads(payload)
    execute.assert_not_called()


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("builtins", "eval"),
        ("builtins", "getattr"),
        ("builtins", "open"),
        ("numpy", "load"),
        ("numpy.rec", "fromfile"),
        ("numpy.core.records", "fromfile"),
        ("numpy._core.records", "fromfile"),
        ("os", "system"),
        ("pandas", "read_pickle"),
        ("pandas.io.pickle", "read_pickle"),
        ("subprocess", "Popen"),
    ],
)
def test_restricted_unpickler_rejects_dangerous_globals(module, name):
    with pytest.raises(pickle.UnpicklingError):
        RestrictedUnpickler(io.BytesIO()).find_class(module, name)


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("numpy", "recarray"),
        ("numpy.rec", "recarray"),
        ("numpy", "record"),
        ("numpy.ma", "MaskedArray"),
        ("numpy.ma.core", "MaskedArray"),
        ("pandas.core.dtypes.dtypes", "SparseDtype"),
        ("pandas.core.arrays.sparse.dtype", "SparseDtype"),
    ],
)
def test_restricted_unpickler_retains_versioned_data_class_paths(module, name):
    # Keep both paths even when the installed version only emits one of them.
    assert (module, name) in get_safe_classes()


def test_restricted_unpickler_rejects_reduce_payload():
    payload = pickle.dumps(_MaliciousPayload())
    with pytest.raises(pickle.UnpicklingError):
        restricted_pickle_loads(payload)


@pytest.mark.parametrize(
    "value",
    [
        np.arange(6).reshape(2, 3),
        pd.Series([1.0, 2.0], index=["a", "b"]),
        pd.DataFrame({"number": [1, 2], "text": ["a", "b"]}),
    ],
)
@pytest.mark.parametrize("protocol", [4, 5])
def test_restricted_unpickler_supports_common_data_objects(value, protocol):
    loaded = restricted_pickle_loads(pickle.dumps(value, protocol=protocol))
    if isinstance(value, np.ndarray):
        np.testing.assert_array_equal(loaded, value)
    elif isinstance(value, pd.Series):
        pd.testing.assert_series_equal(loaded, value)
    else:
        pd.testing.assert_frame_equal(loaded, value)


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize(
    "offset",
    [
        pd.offsets.Day(),
        pd.offsets.BusinessDay(),
        pd.offsets.BusinessHour(),
        pd.offsets.CustomBusinessDay(),
        pd.offsets.Week(),
        pd.offsets.MonthBegin(),
        pd.offsets.MonthEnd(),
        pd.offsets.BMonthBegin(),
        pd.offsets.BMonthEnd(),
        pd.offsets.QuarterBegin(),
        pd.offsets.QuarterEnd(),
        pd.offsets.YearBegin(),
        pd.offsets.YearEnd(),
        pd.offsets.Hour(),
        pd.offsets.Minute(),
        pd.offsets.Second(),
        pd.offsets.Milli(),
        pd.offsets.Micro(),
        pd.offsets.Nano(),
    ],
)
def test_restricted_unpickler_preserves_datetime_frequency(offset, protocol):
    expected = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq=offset))
    actual = restricted_pickle_loads(pickle.dumps(expected, protocol=protocol))
    pd.testing.assert_series_equal(actual, expected)
    assert actual.index.freq == expected.index.freq


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("offset", _BUSINESS_OFFSETS)
@pytest.mark.parametrize("multi_index", [False, True], ids=["datetime-index", "multi-index"])
def test_restricted_unpickler_preserves_business_frequency_frames(offset, multi_index, protocol):
    dates = pd.date_range("2024-01-01 08:30", periods=4, freq=offset, name="datetime")
    index = (
        pd.MultiIndex.from_product([dates, ["SH600000", "SH600004"]], names=["datetime", "instrument"])
        if multi_index
        else dates
    )
    expected = pd.DataFrame(
        {
            "score": np.resize([1.25, np.nan], len(index)),
            "volume": pd.array(np.resize([100, None], len(index)), dtype="Int64"),
        },
        index=index,
    )
    original = expected.copy(deep=True)
    frequency_args = offset.__reduce__()[1]

    actual = restricted_pickle_loads(pickle.dumps(expected, protocol=protocol))

    pd.testing.assert_frame_equal(actual, original, check_exact=True, check_freq=True)
    actual_dates = actual.index.levels[0] if multi_index else actual.index
    pd.testing.assert_index_equal(actual_dates, dates, exact=True, check_exact=True)
    assert type(actual_dates.freq) is type(offset)
    assert actual_dates.freq.__reduce__()[1] == frequency_args
    pd.testing.assert_index_equal(
        pd.date_range(dates[-1], periods=4, freq=actual_dates.freq),
        pd.date_range(dates[-1], periods=4, freq=offset),
        exact=True,
    )
    actual.iloc[0, 0] = -100.0
    pd.testing.assert_frame_equal(expected, original, check_exact=True, check_freq=True)
    assert offset.__reduce__()[1] == frequency_args


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("record_index", [None, 0, 1], ids=["recarray", "record", "missing-record"])
def test_restricted_unpickler_preserves_records(protocol, record_index):
    records = np.array(
        [("SH600000", 1.25, 100, "2024-01-01"), ("SH600004", np.nan, 0, "NaT")],
        dtype=[("instrument", "U8"), ("score", "<f8"), ("volume", "<i4"), ("datetime", "M8[ns]")],
    ).view(np.recarray)
    expected = records if record_index is None else records[record_index]
    original = expected.copy()
    original_bytes = expected.tobytes()

    actual = restricted_pickle_loads(pickle.dumps(expected, protocol=protocol))

    assert type(actual) is (np.recarray if record_index is None else np.record)
    assert actual.dtype == original.dtype
    assert actual.dtype.names == ("instrument", "score", "volume", "datetime")
    assert actual.shape == original.shape
    assert actual.tobytes() == original_bytes
    for name in original.dtype.names:
        np.testing.assert_array_equal(actual[name], original[name])
        np.testing.assert_array_equal(getattr(actual, name), original[name])
    actual["score"] = -100.0
    assert expected.dtype == original.dtype
    assert expected.tobytes() == original_bytes


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("tz", ["UTC", "Asia/Shanghai", "America/New_York"])
def test_restricted_unpickler_preserves_timezone(tz, protocol):
    expected = pd.Series([1, 2], index=pd.date_range("2024-03-10", periods=2, tz=tz))
    actual = restricted_pickle_loads(pickle.dumps(expected, protocol=protocol))
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize(
    "dtype",
    [
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "boolean",
        "string",
    ],
)
def test_restricted_unpickler_preserves_nullable_arrays(dtype, protocol):
    expected = pd.Series([1, None], dtype=dtype)
    actual = restricted_pickle_loads(pickle.dumps(expected, protocol=protocol))
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("value", [pd.NaT, pd.NA])
def test_restricted_unpickler_preserves_missing_scalars(value, protocol):
    assert restricted_pickle_loads(pickle.dumps(value, protocol=protocol)) is value


def test_static_loader_reads_default_pandas_pickle(tmp_path):
    from qlib.data.dataset.loader import StaticDataLoader

    expected = pd.DataFrame({"value": [1.0, 2.0]})
    path = tmp_path / "data.pkl"
    expected.to_pickle(path)
    pd.testing.assert_frame_equal(StaticDataLoader(str(path)).load(), expected)


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("offset", [pd.offsets.Day(), *_BUSINESS_OFFSETS])
def test_restricted_unpickler_rejects_payload_inside_dataframe(protocol, offset, monkeypatch):
    value = pd.DataFrame(
        {"payload": [{"nested": [_MaliciousPayload()]}]},
        index=pd.date_range("2024-01-01", periods=1, freq=offset),
    )
    _assert_rejects_without_execution(value, protocol, monkeypatch)


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("scalar", [False, True], ids=["recarray", "record"])
def test_records_cannot_hide_executable_objects(protocol, scalar, monkeypatch):
    value = np.empty(1, dtype=[("instrument", "U8"), ("payload", object)]).view(np.recarray)
    value.instrument[0] = "SH600000"
    value.payload[0] = {"nested": [_MaliciousPayload()]}
    _assert_rejects_without_execution(value[0] if scalar else value, protocol, monkeypatch)


class _SecondStagePayload:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return pd.read_pickle, (self.path,)


@pytest.mark.parametrize("protocol", [4, 5])
def test_restricted_unpickler_blocks_two_stage_read_pickle(tmp_path, protocol):
    second_stage = tmp_path / "second.pkl"
    second_stage.write_bytes(pickle.dumps(_MaliciousPayload(), protocol=protocol))
    payload = pickle.dumps(_SecondStagePayload(second_stage), protocol=protocol)
    with pytest.raises(pickle.UnpicklingError, match="pandas.*read_pickle"):
        restricted_pickle_loads(payload)


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize(
    "value",
    [
        pd.Series([1, 2], index=pd.period_range("2024-01", periods=2, freq="M")),
        pd.Series(pd.period_range("2024-01", periods=2, freq="M")),
        pd.Series([1, 2], index=pd.IntervalIndex.from_breaks([0, 1, 2])),
        pd.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2])),
        pd.Series([0.0, 1.0, 0.0], dtype=pd.SparseDtype("float64", 0)),
    ],
)
def test_restricted_unpickler_preserves_extended_pandas_types(value, protocol):
    actual = restricted_pickle_loads(pickle.dumps(value, protocol=protocol))
    pd.testing.assert_series_equal(actual, value)


@pytest.mark.parametrize("protocol", [4, 5])
def test_restricted_unpickler_preserves_masked_array(protocol):
    value = np.ma.array([1, 2, 3], mask=[False, True, False])
    actual = restricted_pickle_loads(pickle.dumps(value, protocol=protocol))
    np.testing.assert_array_equal(actual.data, value.data)
    np.testing.assert_array_equal(actual.mask, value.mask)


@pytest.mark.parametrize("protocol", [4, 5])
def test_masked_array_cannot_hide_executable_objects(protocol, monkeypatch):
    value = np.ma.array([_MaliciousPayload()], dtype=object, mask=[True])
    _assert_rejects_without_execution(value, protocol, monkeypatch)
