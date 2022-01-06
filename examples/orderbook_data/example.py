# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import qlib
from qlib.data import D
import unittest

class TestClass(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        provider_uri="~/.qlib/qlib_data/yahoo_cn_1min"
        qlib.init(arctic_uri="127.0.0.1", provider_uri=provider_uri, mem_cache_space_limit=1024**3 * 2, kernels=1)

    def test_basic(self):
        df = D.features(["SH600519"], fields=["@ticks.ask1"], freq="1min", start_time="20201230", end_time="20210101")
        print(df)

    def test_ops1(self):
        df = D.features(["SH600519"], fields=["TResample(@ticks.ask1, '1min', 'last')"], freq="1min", start_time="20201230", end_time="20210101")
        print(df)

    def test_ops2(self):
        df = D.features(["SH600519"], fields=["TRename(TAbs(@ticks.ask1 - TMean(@ticks.ask1, 10)), 'new_name')"], freq="1min", start_time="20201230", end_time="20210101")
        print(df)


if __name__ == "__main__":
    unittest.main()
