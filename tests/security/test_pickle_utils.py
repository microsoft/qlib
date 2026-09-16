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
