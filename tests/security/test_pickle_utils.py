import io
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from qlib.utils.pickle_utils import RestrictedUnpickler, restricted_pickle_loads


class _MaliciousPayload:
    def __reduce__(self):
        return os.system, ("echo vulnerable",)


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("builtins", "eval"),
        ("numpy", "load"),
        ("os", "system"),
        ("pandas", "read_pickle"),
        ("pandas.io.pickle", "read_pickle"),
        ("subprocess", "Popen"),
    ],
)
def test_restricted_unpickler_rejects_dangerous_globals(module, name):
    with pytest.raises(pickle.UnpicklingError):
        RestrictedUnpickler(io.BytesIO()).find_class(module, name)


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
def test_restricted_unpickler_rejects_payload_inside_dataframe(protocol):
    value = pd.DataFrame({"payload": [_MaliciousPayload()]})
    with pytest.raises(pickle.UnpicklingError, match="Forbidden class"):
        restricted_pickle_loads(pickle.dumps(value, protocol=protocol))


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
def test_masked_array_cannot_hide_executable_objects(protocol):
    value = np.ma.array([_MaliciousPayload()], dtype=object, mask=[True])
    with pytest.raises(pickle.UnpicklingError, match="Forbidden class"):
        restricted_pickle_loads(pickle.dumps(value, protocol=protocol))
