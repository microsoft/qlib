# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from qlib.data import D
from qlib.tests import TestAutoData
from qlib.data.dataset.processor import MinMaxNorm, ZScoreNorm, CSZScoreNorm, CSZFillna


class TestProcessor(TestAutoData):
    TEST_INST = "SH600519"

    def test_MinMaxNorm(self):
        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        mmn = MinMaxNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        mmn.fit(df)
        mmn.__call__(df)
        assert (df.tail(5).iloc[:, :-1] != origin_df.tail(5).iloc[:, :-1]).all().all()

    def test_ZScoreNorm(self):
        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        zsn = ZScoreNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        zsn.fit(df)
        zsn.__call__(df)
        assert (df.tail(5).iloc[:, :-1] != origin_df.tail(5).iloc[:, :-1]).all().all()

    def test_CSZFillna(self):
        origin_df = D.features([self.TEST_INST], fields=["$high", "$open", "$low", "$close"])[113:118]
        df = origin_df.copy()
        CSZFillna(fields_group=None).__call__(df)
        assert ~((origin_df == df)[1:2].all().all())

    def test_CSZScoreNorm(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[10:12])[50:70]
        df = origin_df.copy()
        CSZScoreNorm(fields_group=None).__call__(df)
        assert (df[2:4] == ((origin_df[2:4] - origin_df[2:4].mean()).div(origin_df[2:4].std()))).all().all()


if __name__ == "__main__":
    unittest.main()
