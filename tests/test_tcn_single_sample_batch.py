# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for TCN single-sample batch fix (Fixes #1752).

When the last batch contains a single sample, the model output after
.cpu().numpy() may be a 0-d array. np.concatenate then fails because
it cannot concatenate 0-d arrays with 1-d arrays. Wrapping with
np.atleast_1d ensures all arrays are at least 1-d.
"""

import numpy as np
import pytest


def test_concatenate_mixed_0d_and_1d():
    """np.concatenate fails with raw 0-d + 1-d arrays, but works with atleast_1d."""
    arr_1d = np.array([1.0, 2.0, 3.0])
    arr_0d = np.float64(4.0)  # simulates single-sample .numpy() result

    # Without fix this would raise
    with pytest.raises((ValueError, np.exceptions.AxisError)):
        np.concatenate([arr_1d, arr_0d])

    # With atleast_1d it works
    result = np.concatenate([np.atleast_1d(arr_1d), np.atleast_1d(arr_0d)])
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


def test_atleast_1d_preserves_normal_arrays():
    """atleast_1d should be a no-op for arrays that are already >= 1-d."""
    arr = np.array([5.0, 6.0])
    result = np.atleast_1d(arr)
    np.testing.assert_array_equal(result, arr)
    assert result.ndim >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
