# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import unittest
import pandas as pd
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
        st = """
        2000-01-01,1,2
        2000-01-02,,4
        2000-01-03,5,6
        """
        origin_df = pd.read_csv(io.StringIO(st), header=None)
        origin_df.columns = ["datetime", "a", "b"]
        origin_df.set_index("datetime", inplace=True, drop=True)
        df = origin_df.copy()
        CSZFillna(fields_group=None).__call__(df)
        assert ~((origin_df == df).iloc[1, 0])

    def test_CSZScoreNorm(self):
        st = """
        2000-01-01,1,2
        2000-01-02,3,4
        2000-01-03,5,6
        """
        origin_df = pd.read_csv(io.StringIO(st), header=None)
        origin_df.columns = ["datetime", "a", "b"]
        origin_df.set_index("datetime", inplace=True, drop=True)
        df = origin_df.copy()
        CSZScoreNorm(fields_group=None).__call__(df)
        assert (df == ((origin_df - origin_df.mean()).div(origin_df.std()))).all().all()


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestProcessor("test_MinMaxNorm"))
    _suite.addTest(TestProcessor("test_ZScoreNorm"))
    _suite.addTest(TestProcessor("test_CSZFillna"))
    _suite.addTest(TestProcessor("test_CSZScoreNorm"))
    return _suite
