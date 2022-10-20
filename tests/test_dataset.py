# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import pytest
import sys
from qlib.tests import TestAutoData
from qlib.data.dataset import TSDatasetH
import numpy as np
import time
from qlib.data.dataset.handler import DataHandlerLP


class TestDataset(TestAutoData):
    @pytest.mark.slow
    def testTSDataset(self):
        tsdh = TSDatasetH(
            handler={
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2017-01-01",
                    "end_time": "2020-08-01",
                    "fit_start_time": "2017-01-01",
                    "fit_end_time": "2017-12-31",
                    "instruments": "csi300",
                    "infer_processors": [
                        {"class": "FilterCol", "kwargs": {"col_list": ["RESI5", "WVMA5", "RSQR5"]}},
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": "true"}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [
                        "DropnaLabel",
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},  # CSRankNorm
                    ],
                },
            },
            segments={
                "train": ("2017-01-01", "2017-12-31"),
                "valid": ("2018-01-01", "2018-12-31"),
                "test": ("2019-01-01", "2020-08-01"),
            },
        )
        tsds_train = tsdh.prepare("train", data_key=DataHandlerLP.DK_L)  # Test the correctness
        tsds = tsdh.prepare("valid", data_key=DataHandlerLP.DK_L)

        t = time.time()
        for idx in np.random.randint(0, len(tsds_train), size=2000):
            _ = tsds_train[idx]
        print(f"2000 sample takes {time.time() - t}s")

        t = time.time()
        for _ in range(20):
            data = tsds_train[np.random.randint(0, len(tsds_train), size=2000)]
        print(data.shape)
        print(f"2000 sample(batch index) * 20 times takes {time.time() - t}s")

        # The dimension of sample is same as tabular data, but it will return timeseries data of the sample

        # We have two method to get the time-series of a sample

        # 1) sample by int index directly
        tsds[len(tsds) - 1]

        # 2) sample by <datetime,instrument> index
        data_from_ds = tsds["2017-12-31", "SZ300315"]

        # Check the data
        # Get data from DataFrame Directly
        data_from_df = (
            tsdh.handler.fetch(data_key=DataHandlerLP.DK_L)
            .loc(axis=0)["2017-01-01":"2017-12-31", "SZ300315"]
            .iloc[-30:]
            .values
        )

        equal = np.isclose(data_from_df, data_from_ds)
        self.assertTrue(equal[~np.isnan(data_from_df)].all())

        if False:
            # 3) get both index and data
            # NOTE: We don't want to reply on pytorch, so this test can't be included. It is just a example
            from torch.utils.data import DataLoader
            from qlib.model.utils import IndexSampler

            i = len(tsds) - 1
            idx = tsds.get_index()
            tsds[i]
            idx[i]

            s_w_i = IndexSampler(tsds)
            test_loader = DataLoader(s_w_i)

            s_w_i[3]
            for data, i in test_loader:
                break
            print(data.shape)
            print(idx[i])


if __name__ == "__main__":
    unittest.main(verbosity=10)

    # User could use following code to run test when using line_profiler
    # td = TestDataset()
    # td.setUpClass()
    # td.testTSDataset()
