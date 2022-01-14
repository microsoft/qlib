# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from arctic.arctic import Arctic
import qlib
from qlib.data import D
import unittest


class TestClass(unittest.TestCase):
    """
    Useful commands
    - run all tests: pytest examples/orderbook_data/example.py
    - run a single test:  pytest -s --pdb --disable-warnings examples/orderbook_data/example.py::TestClass::test_basic01
    """

    def setUp(self):
        """
        Configure for arctic
        """
        provider_uri = "~/.qlib/qlib_data/yahoo_cn_1min"
        qlib.init(
            provider_uri=provider_uri,
            mem_cache_size_limit=1024 ** 3 * 2,
            mem_cache_type="sizeof",
            kernels=1,
            expression_provider={"class": "LocalExpressionProvider", "kwargs": {"time2idx": False}},
            feature_provider={"class": "ArcticFeatureProvider", "kwargs": {"uri": "127.0.0.1"}},
            dataset_provider={
                "class": "LocalDatasetProvider",
                "kwargs": {
                    "align_time": False,  # Order book is not fixed, so it can't be align to a shared fixed frequency calendar
                },
            },
        )

    def test_basic(self):
        df = D.features(["SH600519"], fields=["$ask1"], freq="ticks", start_time="20201230", end_time="20210101")
        print(df)

    def test_basic_without_time(self):
        df = D.features(["SH600519"], fields=["$ask1"], freq="ticks")
        print(df)

    def test_basic01(self):
        df = D.features(
            ["SH600519"],
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)



if __name__ == "__main__":
    unittest.main()
