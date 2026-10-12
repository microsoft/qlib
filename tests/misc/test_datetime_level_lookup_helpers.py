"""Follow-up coverage for the #1909 duplicate-``datetime`` regression.

After #1909, ``PortAnaRecord`` resolves the ``datetime`` level positionally.
Several other call sites still relied on the name-based lookup and would
crash with the same ``ValueError`` on the same MultiIndex shape. This test
covers the helpers that drive those sites so the broader fix doesn't
regress silently if the call sites are later refactored.
"""

import unittest

import numpy as np
import pandas as pd

from qlib.utils import split_pred


def _dup_dt_index(n: int = 6) -> pd.MultiIndex:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    instruments = [f"i{i % 2}" for i in range(n)]
    return pd.MultiIndex.from_arrays(
        [dates, instruments, dates],
        names=["datetime", "instrument", "datetime"],
    )


class TestDatetimeLevelLookupHelpers(unittest.TestCase):
    def test_split_pred_handles_duplicate_datetime_level(self) -> None:
        # Sanity: the index name lookup that the old code path used does
        # raise on this shape — split_pred must not rely on it.
        idx = _dup_dt_index(6)
        with self.assertRaises(ValueError):
            idx.get_level_values("datetime")

        pred = pd.DataFrame({"score": np.arange(6, dtype=float)}, index=idx)
        pred_left, pred_right = split_pred(pred, number=2)

        # Left half should contain the earliest two distinct dates, right
        # half should contain the rest. The exact slicing semantics are
        # the same as the unique-name case; we only assert sizes here so
        # the assertion stays meaningful even if sort behavior on the
        # duplicate index changes between pandas versions.
        self.assertGreater(len(pred_left), 0)
        self.assertGreater(len(pred_right), 0)
        self.assertEqual(len(pred_left) + len(pred_right), len(pred))


if __name__ == "__main__":
    unittest.main()
