import numpy as np
import pandas as pd
import pytest

from qlib.data.base import Feature
from qlib.data.ops import IdxMax, IdxMin


class SeriesFeature(Feature):
    """Leaf feature backed by a fixed series for operator unit tests."""

    def __init__(self, values):
        super().__init__("series")
        self._values = values

    def _load_internal(self, instrument, start_index, end_index, *args):
        return pd.Series(self._values, dtype="float64")


@pytest.mark.parametrize(
    ("operator", "window", "expected"),
    [
        (IdxMax, 3, [1.0, 2.0, 2.0, 1.0, 3.0]),
        (IdxMin, 3, [1.0, 1.0, 1.0, 3.0, 2.0]),
        (IdxMax, 0, [1.0, 2.0, 2.0, 2.0, 2.0]),
        (IdxMin, 0, [1.0, 1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_idx_extrema_ignore_nan_in_window(operator, window, expected):
    feature = SeriesFeature([1.0, 5.0, np.nan, 2.0, 3.0])

    result = operator(feature, window)._load_internal("TEST", 0, 4, "day")

    np.testing.assert_allclose(result.to_numpy(), expected)


@pytest.mark.parametrize("operator", [IdxMax, IdxMin])
def test_idx_extrema_return_nan_for_all_nan_windows(operator):
    feature = SeriesFeature([np.nan, np.nan])

    result = operator(feature, 2)._load_internal("TEST", 0, 1, "day")

    assert result.isna().all()
