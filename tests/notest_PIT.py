# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import unittest
import qlib
from qlib.data import D
from qlib.tests import TestAutoData
from qlib.config import REG_CN


class TestRegiterCustomOps(TestAutoData):
    @classmethod
    def setUpClass(cls) -> None:
        # use default data
        provider_uri = "~/.qlib/qlib_data/cn_data_new"  # target_dir
        qlib.init(provider_uri=provider_uri, region=REG_CN)

    def test_regiter_custom_ops(self):

        instruments = ["sz000708"]
        fields = ["$$roewa_q", "$$yoyni_q"]
        fields += ["($$roewa_q / $$yoyni_q) / PRef($$roewa_q / $$yoyni_q, 1) - 1"]
        fields += ["PSum($$yoyni_q, 4)"]
        fields += ["$close", "$$roewa_q*$close"]
        print(D.features(instruments, fields, start_time="2019-01-01", end_time="2020-01-01", freq="day"))


if __name__ == "__main__":
    unittest.main()
