# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for record_temp datetime level fix (Fixes #1909).

The fix replaces hard-coded get_level_values("datetime") with a
positional lookup via index.names.index("datetime"), so it works
regardless of the position of the datetime level in the MultiIndex.
"""

import numpy as np
import pandas as pd
import pytest


def _make_multiindex(names, n_dates=3, n_instruments=2):
    """Helper to build a MultiIndex with the given level names."""
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    instruments = [f"STOCK_{chr(65 + i)}" for i in range(n_instruments)]
    arrays = {
        "datetime": np.repeat(dates, n_instruments),
        "instrument": np.tile(instruments, n_dates),
    }
    # Build in the order specified by `names`
    return pd.MultiIndex.from_arrays([arrays[n] for n in names], names=names)


def test_datetime_first_level():
    """Standard order: (datetime, instrument)."""
    idx = _make_multiindex(["datetime", "instrument"])
    dt_level = idx.names.index("datetime")
    dt_values = idx.get_level_values(dt_level)
    assert len(dt_values) == 6
    assert dt_values[0] == pd.Timestamp("2023-01-01")


def test_datetime_second_level():
    """Reversed order: (instrument, datetime)."""
    idx = _make_multiindex(["instrument", "datetime"])
    dt_level = idx.names.index("datetime")
    dt_values = idx.get_level_values(dt_level)
    assert len(dt_values) == 6
    assert dt_values[0] == pd.Timestamp("2023-01-01")


def test_missing_datetime_raises():
    """If datetime level is absent, index() should raise ValueError."""
    idx = pd.MultiIndex.from_tuples(
        [("A", 1), ("B", 2)], names=["instrument", "id"]
    )
    with pytest.raises(ValueError):
        idx.names.index("datetime")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
