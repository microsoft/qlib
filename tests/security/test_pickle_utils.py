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
def test_restricted_unpickler_supports_common_data_objects(value):
    loaded = restricted_pickle_loads(pickle.dumps(value, protocol=4))
    if isinstance(value, np.ndarray):
        np.testing.assert_array_equal(loaded, value)
    elif isinstance(value, pd.Series):
        pd.testing.assert_series_equal(loaded, value)
    else:
        pd.testing.assert_frame_equal(loaded, value)
