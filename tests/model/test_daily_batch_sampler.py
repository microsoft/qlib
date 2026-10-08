"""Regression tests for DailyBatchSampler (microsoft/qlib#2319).

TSDataSampler stores rows instrument-major (<instrument, datetime>);
get_index() swaps the label order only. DailyBatchSampler used to slice
contiguous ranges by per-day counts, which produced cross-day batches of a
single instrument. Each yielded batch must be one trading day's full
cross-section instead.
"""

import unittest

import numpy as np
import pandas as pd

from qlib.contrib.model.pytorch_gats_ts import DailyBatchSampler


class _FakeDataSource:
    """Mirror of TSDataSampler.get_index(): stores rows instrument-major
    (<instrument, datetime>) and returns the SWAPPED label view
    (<datetime, instrument>) without reordering the rows.
    """

    def __init__(self, data_index: pd.MultiIndex):
        self._data_index = data_index

    def get_index(self) -> pd.MultiIndex:
        return self._data_index.swaplevel()


def _instrument_major_index(instruments, dates) -> pd.MultiIndex:
    """Rows physically sorted <instrument, datetime>, like TSDataSampler.data_index."""
    rows = [(inst, day) for inst in instruments for day in dates]
    return pd.MultiIndex.from_tuples(rows, names=["instrument", "datetime"])


class TestDailyBatchSampler(unittest.TestCase):
    def test_batches_are_per_day_cross_sections(self):
        instruments = ["SH600000", "SH600008", "SH600009"]
        dates = list(pd.date_range("2026-01-05", periods=4, freq="B"))
        index = _instrument_major_index(instruments, dates)

        sampler = DailyBatchSampler(_FakeDataSource(index))
        batches = list(iter(sampler))

        # one batch per trading day
        self.assertEqual(len(batches), len(dates))

        covered = []
        for batch in batches:
            labels = index[batch]
            datetimes = {day for _, day in labels}
            # every batch covers exactly one trading day ...
            self.assertEqual(len(datetimes), 1)
            # ... and that day's full instrument cross-section
            self.assertEqual(len(labels), len(instruments))
            covered.extend(batch.tolist())

        # every physical row is yielded exactly once
        self.assertEqual(sorted(covered), list(range(len(index))))

    def test_batches_follow_chronological_day_order(self):
        instruments = ["SH600000", "SH600008"]
        dates = list(pd.date_range("2026-01-05", periods=3, freq="B"))
        index = _instrument_major_index(instruments, dates)

        sampler = DailyBatchSampler(_FakeDataSource(index))
        batches = list(iter(sampler))

        first_days = []
        for batch in batches:
            labels = index[batch]
            first_days.append(min(day for _, day in labels))
        self.assertEqual(first_days, dates)

    def test_len_matches_number_of_batches(self):
        instruments = ["SH600000", "SH600008"]
        dates = list(pd.date_range("2026-01-05", periods=3, freq="B"))
        index = _instrument_major_index(instruments, dates)

        sampler = DailyBatchSampler(_FakeDataSource(index))
        self.assertEqual(len(sampler), len(dates))


if __name__ == "__main__":
    unittest.main()
