# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.contrib.eva.alpha import calc_group_returns


def _panel(scores, labels, dates=None):
    dates = pd.to_datetime(dates or ["2020-01-02"])
    names = [f"S{i}" for i in range(len(scores))]
    index = pd.MultiIndex.from_product([dates, names], names=["datetime", "instrument"])
    return pd.DataFrame({"score": scores, "label": labels}, index=index)


class TestFactorGroupReturn(unittest.TestCase):
    def test_last_group_keeps_the_lowest_scores(self):
        # 7 names and 5 groups used to leave the two worst names out of Group5.
        scores = [1, 7, 3, 5, 2, 6, 4]
        labels = scores
        out = calc_group_returns(_panel(scores, labels), n_groups=5)

        self.assertAlmostEqual(out.loc[pd.Timestamp("2020-01-02"), "Group1"], 7)
        self.assertAlmostEqual(out.loc[pd.Timestamp("2020-01-02"), "Group5"], np.mean([3, 2, 1]))
        self.assertAlmostEqual(out.loc[pd.Timestamp("2020-01-02"), "long-short"], 7 - np.mean([3, 2, 1]))

    def test_even_split_is_unchanged(self):
        scores = list(range(1, 11))
        out = calc_group_returns(_panel(scores, scores), n_groups=5)
        day = out.loc[pd.Timestamp("2020-01-02")]

        self.assertAlmostEqual(day["Group1"], np.mean([10, 9]))
        self.assertAlmostEqual(day["Group5"], np.mean([2, 1]))
        self.assertAlmostEqual(day["long-short"], np.mean([10, 9]) - np.mean([2, 1]))
        self.assertAlmostEqual(day["long-average"], np.mean([10, 9]) - np.mean(scores))

    def test_missing_score_stays_in_the_cross_section_average(self):
        scores = [5, 4, 3, 2, np.nan]
        labels = [1, 1, 1, 1, 10]
        out = calc_group_returns(_panel(scores, labels), n_groups=2)
        day = out.loc[pd.Timestamp("2020-01-02")]

        self.assertAlmostEqual(day["Group1"], 1)
        self.assertAlmostEqual(day["Group2"], 1)
        self.assertAlmostEqual(day["long-average"], 1 - np.mean(labels))

    def test_fewer_names_than_groups_is_all_nan(self):
        out = calc_group_returns(_panel([3, 1, 2], [3, 1, 2]), n_groups=5)
        day = out.loc[pd.Timestamp("2020-01-02")]
        for column in ("Group1", "Group2", "Group3", "Group4", "Group5", "long-short", "long-average"):
            self.assertTrue(np.isnan(day[column]), column)

    def test_n_groups_below_two_is_rejected(self):
        with self.assertRaises(ValueError):
            calc_group_returns(_panel([1, 2], [1, 2]), n_groups=1)


if __name__ == "__main__":
    unittest.main()
