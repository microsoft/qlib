# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import sys
from qlib.tests import TestAutoData
from qlib.data.dataset import TSDatasetH
import numpy as np


class TestDataset(TestAutoData):
    def testTSDataset(self):
        tsdh = TSDatasetH(
            handler={
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2008-01-01",
                    "end_time": "2020-08-01",
                    "fit_start_time": "2008-01-01",
                    "fit_end_time": "2014-12-31",
                    "instruments": "csi300",
                },
            },
            segments={
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        )
        _ = tsdh.prepare("train")  # Test the correctness
        tsds = tsdh.prepare("valid")  # prepare a dataset with is friendly to converting tabular data to time-series

        # The dimension of sample is same as tabular data, but it will return timeseries data of the sample

        # We have two method to get the time-series of a sample

        # 1) sample by int index directly
        tsds[len(tsds) - 1]

        # 2) sample by <datetime,instrument> index
        data_from_ds = tsds["2016-12-31", "SZ300315"]

        # Check the data
        # Get data from DataFrame Directly
        data_from_df = tsdh._handler.fetch().loc(axis=0)["2015-01-01":"2016-12-31", "SZ300315"].iloc[-30:].values

        equal = np.isclose(data_from_df, data_from_ds)
        self.assertTrue(equal[~np.isnan(data_from_df)].all())


if __name__ == "__main__":
    unittest.main(verbosity=10)
