# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import numpy as np
from qlib.data import D
from qlib.tests import TestAutoData
from qlib.data.dataset.processor import MinMaxNorm, ZScoreNorm, CSZScoreNorm, CSZFillna


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
