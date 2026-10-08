import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from qlib.utils.resam import get_higher_eq_freq_feature


class TestGetHigherEqFreqFeature(unittest.TestCase):
    """Regression test for https://github.com/microsoft/qlib/issues/1925

    `time_per_step="1day"` (as used by some executors) and `freq="day"` are the
    same frequency, but were previously passed to `D.features` verbatim. Since
    feature files on disk are named after the *normalized* freq (e.g. `day`,
    not `1day`), an un-normalized `1day` request silently found no data instead
    of matching the existing `day` data.
    """

    def test_freq_is_normalized_before_query(self):
        mock_d = MagicMock()
        mock_d.features.return_value = pd.DataFrame({"$close": [1.0]})
        with patch("qlib.data.data.D", mock_d):
            _, used_freq = get_higher_eq_freq_feature(["SH000300"], ["$close"], freq="1day")

        self.assertEqual(used_freq, "day")
        self.assertEqual(mock_d.features.call_args.kwargs["freq"], "day")

    def test_already_normalized_freq_is_unchanged(self):
        mock_d = MagicMock()
        mock_d.features.return_value = pd.DataFrame({"$close": [1.0]})
        with patch("qlib.data.data.D", mock_d):
            _, used_freq = get_higher_eq_freq_feature(["SH000300"], ["$close"], freq="day")

        self.assertEqual(used_freq, "day")
        self.assertEqual(mock_d.features.call_args.kwargs["freq"], "day")


if __name__ == "__main__":
    unittest.main()
