# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch

import pandas as pd

from qlib.backtest.report import PortfolioMetrics


class TestBenchmarkWeighting:
    @patch("qlib.backtest.report.get_higher_eq_freq_feature")
    def test_list_benchmark_is_equal_weighted(self, get_feature):
        dates = pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-05"])
        instruments = ["A", "B", "A", "B"]
        index = pd.MultiIndex.from_arrays(
            [instruments, dates], names=["instrument", "datetime"]
        )
        returns = pd.DataFrame(
            {"$close/Ref($close,1)-1": [0.10, 0.00, -0.05, 0.04]},
            index=index,
        )
        get_feature.return_value = (returns, None)

        benchmark = PortfolioMetrics._cal_benchmark(
            {"benchmark": ["A", "B"]}, freq="day"
        )

        expected = pd.Series(
            [0.05, -0.005],
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
            name="$close/Ref($close,1)-1",
        )
        expected.index.name = "datetime"
        pd.testing.assert_series_equal(benchmark, expected)
