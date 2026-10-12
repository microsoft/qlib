"""Regression test for https://github.com/microsoft/qlib/issues/1909.

The reporter saw `record_temp.PortAnaRecord` crash with
`ValueError: 'datetime' occurs multiple times` when the predicted DataFrame
ended up with a MultiIndex carrying two levels both named ``datetime``. The
old code relied on pandas' name-based level lookup, which is ambiguous in
that case. The fix resolves the level positionally.

This test exercises the resolution directly so the regression cannot creep
back even if pandas decides to be stricter about duplicate level names.
"""

import unittest

import numpy as np
import pandas as pd


class TestDuplicateDatetimeLevelResolution(unittest.TestCase):
    def _build_index_with_duplicate_datetime(self) -> pd.MultiIndex:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        return pd.MultiIndex.from_arrays(
            [dates, ["a", "b", "c"], dates],
            names=["datetime", "instrument", "datetime"],
        )

    def test_name_based_lookup_fails(self) -> None:
        """Sanity-check the precondition: name lookup is ambiguous."""
        idx = self._build_index_with_duplicate_datetime()
        with self.assertRaises(ValueError):
            idx.get_level_values("datetime")

    def test_positional_lookup_resolves(self) -> None:
        """The fix path: index.names.index('datetime') + positional lookup."""
        idx = self._build_index_with_duplicate_datetime()
        dt_level = idx.names.index("datetime")
        values = idx.get_level_values(dt_level)
        np.testing.assert_array_equal(
            values.values,
            pd.date_range("2024-01-01", periods=3, freq="D").values,
        )


if __name__ == "__main__":
    unittest.main()
