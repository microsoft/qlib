# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import unittest
import qlib
from qlib.data import D
from qlib.tests import TestAutoData


class TestRegiterCustomOps(TestAutoData):
    @classmethod
    def setUpClass(cls) -> None:
        # use default data
        qlib.init()

    def test_regiter_custom_ops(self):

        instruments = ["sh600519"]
        fields = ["$$roewa_q", "$$yoyni_q"]
        fields += ["($$roewa_q / $$yoyni_q) / PRef($$roewa_q / $$yoyni_q, 1) - 1"]
        fields += ["PSum($$yoyni_q, 4)"]
        fields += ["$close", "$$roewa_q*$close"]
        data = D.features(instruments, fields, start_time="2019-01-01", end_time="2020-01-01", freq="day")
        print(data)
        print(data.describe())


if __name__ == "__main__":
    unittest.main()
