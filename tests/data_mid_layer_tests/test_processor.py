# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import numpy as np
import pandas as pd
from qlib.data import D
from qlib.tests import TestAutoData
from qlib.data.dataset.processor import MinMaxNorm, ZScoreNorm, CSZScoreNorm, CSZFillna, TanhProcess


class TestTanhProcess(unittest.TestCase):
    """Unit tests for TanhProcess (issue #1687).

    These tests do not require downloaded market data.
    """

    @staticmethod
    def _make_handler_like_df():
        # Match Alpha158/Alpha360 MultiIndex layout:
        # level 0 = group (feature/label), level 1 = column name.
        columns = pd.MultiIndex.from_tuples(
            [("feature", "RESI5"), ("feature", "WVMA5"), ("label", "LABEL0")],
            names=["field", "name"],
        )
        index = pd.MultiIndex.from_tuples(
            [("2023-01-01", "SH600000"), ("2023-01-02", "SH600000")],
            names=["datetime", "instrument"],
        )
        data = np.array([[2.0, 3.0, 0.5], [4.0, 5.0, 0.8]], dtype=float)
        return pd.DataFrame(data, index=index, columns=columns)

    def test_default_transforms_feature_group_only(self):
        df = self._make_handler_like_df()
        label_before = df[("label", "LABEL0")].copy()

        result = TanhProcess()(df)

        pd.testing.assert_series_equal(result[("label", "LABEL0")], label_before)
        np.testing.assert_allclose(
            result[("feature", "RESI5")].to_numpy(),
            np.tanh(np.array([2.0, 4.0]) - 1),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            result[("feature", "WVMA5")].to_numpy(),
            np.tanh(np.array([3.0, 5.0]) - 1),
            atol=1e-7,
        )

    def test_fields_group_none_transforms_all_columns(self):
        df = self._make_handler_like_df()
        expected = np.tanh(df.to_numpy() - 1)

        result = TanhProcess(fields_group=None)(df)

        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-7)

    def test_legacy_label_substring_mask_was_incorrect_for_standard_groups(self):
        # Regression: matching "LABEL" on level 1 happens to catch LABEL0, but
        # matching on level 0 (as suggested in some reports) would miss lowercase
        # "label" and transform labels. fields_group avoids both pitfalls.
        df = self._make_handler_like_df()
        level0_mask = df.columns.get_level_values(0).str.contains("LABEL")
        self.assertFalse(level0_mask.any())

        label_before = df[("label", "LABEL0")].copy()
        TanhProcess(fields_group="feature")(df)
        pd.testing.assert_series_equal(df[("label", "LABEL0")], label_before)


class TestProcessor(TestAutoData):
    TEST_INST = "SH600519"

    def test_MinMaxNorm(self):
        def normalize(df):
            min_val = np.nanmin(df.values, axis=0)
            max_val = np.nanmax(df.values, axis=0)
            ignore = min_val == max_val
            for _i, _con in enumerate(ignore):
                if _con:
                    max_val[_i] = 1
                    min_val[_i] = 0
            df.loc(axis=1)[df.columns] = (df.values - min_val) / (max_val - min_val)
            return df

        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        mmn = MinMaxNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        mmn.fit(df)
        mmn.__call__(df)
        origin_df = normalize(origin_df)
        assert (df == origin_df).all().all()

    def test_ZScoreNorm(self):
        def normalize(df):
            mean_train = np.nanmean(df.values, axis=0)
            std_train = np.nanstd(df.values, axis=0)
            ignore = std_train == 0
            for _i, _con in enumerate(ignore):
                if _con:
                    std_train[_i] = 1
                    mean_train[_i] = 0
            df.loc(axis=1)[df.columns] = (df.values - mean_train) / std_train
            return df

        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        zsn = ZScoreNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        zsn.fit(df)
        zsn.__call__(df)
        origin_df = normalize(origin_df)
        assert (df == origin_df).all().all()

    def test_CSZFillna(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[97:99])[228:238]
        df = origin_df.copy()
        CSZFillna(fields_group=None).__call__(df)
        assert ~df[1:2].isna().all().all() and origin_df[1:2].isna().all().all()

    def test_CSZScoreNorm(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[10:12])[50:60]
        df = origin_df.copy()
        CSZScoreNorm(fields_group=None).__call__(df)
        # If we use the formula directly on the original data, we cannot get the correct result,
        # because the original data is processed by `groupby`, so we use the method of slicing,
        # taking the 2nd group of data from the original data, to calculate and compare.
        assert (df[2:4] == ((origin_df[2:4] - origin_df[2:4].mean()).div(origin_df[2:4].std()))).all().all()


if __name__ == "__main__":
    unittest.main()
