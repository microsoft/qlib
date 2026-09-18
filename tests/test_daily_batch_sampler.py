# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd

from qlib.contrib.model.pytorch_gats_ts import DailyBatchSampler
from qlib.data.dataset import TSDataSampler


class TestDailyBatchSampler(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        instruments = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        data = pd.DataFrame({"feature": np.arange(len(index)), "label": np.arange(len(index))}, index=index)
        self.data_source = TSDataSampler(data, dates[0], dates[-1], step_len=2)

    def test_each_batch_is_one_day(self):
        sampler = DailyBatchSampler(self.data_source)
        index = self.data_source.get_index()
        for batch in sampler:
            batch_index = index[batch]
            self.assertEqual(batch_index.get_level_values("datetime").nunique(), 1)
            self.assertEqual(batch_index.get_level_values("instrument").nunique(), 3)

    def test_batches_cover_all_samples_once(self):
        sampler = DailyBatchSampler(self.data_source)
        positions = np.concatenate(list(sampler))
        self.assertEqual(sorted(positions.tolist()), list(range(len(self.data_source.get_index()))))
        self.assertEqual(len(sampler), 5)


if __name__ == "__main__":
    unittest.main()
