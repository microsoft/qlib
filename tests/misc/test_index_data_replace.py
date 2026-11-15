# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest

from qlib.utils.index_data import MultiData, SingleData


@pytest.mark.parametrize("nan_key", [np.nan, float("nan"), np.float32("nan"), np.float64("nan")])
@pytest.mark.parametrize("multidimensional", [False, True])
def test_replace_nan_preserves_values_indices_and_input(nan_key, multidimensional):
    values = np.array([np.nan, 1.0, -2.0, np.nan, np.inf, -np.inf])
    if multidimensional:
        values = values.reshape(2, 3)
        data = MultiData(values, index=["b", "a"], columns=["z", "x", "y"])
    else:
        data = SingleData(values, index=["f", "e", "d", "c", "b", "a"])
    result = data.replace({nan_key: 0})
    expected = np.where(np.isnan(values), 0, values)
    np.testing.assert_array_equal(result.data, expected)
    np.testing.assert_array_equal(data.data, values)
    assert type(result) is type(data)
    assert not np.shares_memory(result.data, data.data)
    for original_index, result_index in zip(data.indices, result.indices):
        assert original_index == result_index


@pytest.mark.parametrize(
    "values,replacements,expected",
    [
        ([np.nan, np.nan], {np.nan: 0}, [0, 0]),
        ([1, 2], {np.nan: 0}, [1, 2]),
        ([np.nan, 1], {}, [np.nan, 1]),
        ([np.nan, 1, 2], {1: np.nan}, [np.nan, np.nan, 2]),
        ([np.nan, 1, 2], {np.nan: 0, 1: np.nan}, [0, np.nan, 2]),
        ([np.nan, 1, 2], {1: np.nan, np.nan: 0}, [0, np.nan, 2]),
        ([np.nan, 1, 2], {np.nan: np.nan, 1: 2, 2: 1}, [np.nan, 2, 1]),
        ([1, 2, 3], {1: 2, 2: 3}, [2, 3, 3]),
        ([np.inf, -np.inf, np.nan], {np.inf: 1, -np.inf: -1}, [1, -1, np.nan]),
        ([], {np.nan: 0}, []),
    ],
)
def test_replace_uses_original_values(values, replacements, expected):
    data = SingleData(values)
    result = data.replace(replacements)
    np.testing.assert_array_equal(result.data, expected)
    np.testing.assert_array_equal(data.data, values)
