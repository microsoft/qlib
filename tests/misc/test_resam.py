# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from qlib.constant import REG_CN
from qlib.utils.resam import resam_calendar


@pytest.mark.parametrize(
    "dates, raw_freq, sampled_freq, expected",
    [
        (
            ["2024-01-01", "2024-01-02", "2024-01-09", "2024-01-10", "2024-01-17"],
            "day",
            "week",
            ["2024-01-01", "2024-01-09", "2024-01-17"],
        ),
        (
            ["2023-12-25", "2024-01-01", "2024-01-08", "2024-01-15"],
            "week",
            "2week",
            ["2023-12-25", "2024-01-08"],
        ),
        (
            ["2023-12-04", "2023-12-05", "2024-01-08", "2024-01-09", "2024-02-12"],
            "day",
            "month",
            ["2023-12-04", "2024-01-08", "2024-02-12"],
        ),
        (
            ["2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01"],
            "month",
            "2month",
            ["2023-12-01", "2024-02-01"],
        ),
        (
            ["2023-12-29 09:30", "2023-12-29 13:00", "2024-01-02 09:30", "2024-01-02 13:00"],
            "1min",
            "week",
            ["2023-12-29", "2024-01-02"],
        ),
    ],
)
def test_resam_calendar_period_boundaries(dates, raw_freq, sampled_freq, expected):
    calendar = np.array([pd.Timestamp(value) for value in dates])
    result = resam_calendar(calendar, raw_freq, sampled_freq, region=REG_CN)
    np.testing.assert_array_equal(result, np.array([pd.Timestamp(value) for value in expected]))


@pytest.mark.parametrize(
    "start, end, gap_start, gap_end, sampled_freq, expected",
    [
        (
            "2024-01-29",
            "2024-03-08",
            "2024-02-12",
            "2024-02-16",
            "2week",
            ["2024-01-29", "2024-02-19", "2024-03-04"],
        ),
        (
            "2024-01-01",
            "2024-06-28",
            "2024-03-01",
            "2024-03-31",
            "2month",
            ["2024-01-01", "2024-04-01", "2024-06-03"],
        ),
    ],
)
def test_resam_calendar_counts_nonempty_periods(start, end, gap_start, gap_end, sampled_freq, expected):
    calendar = pd.bdate_range(start, end)
    calendar = calendar[(calendar < gap_start) | (calendar > gap_end)]
    result = resam_calendar(calendar.to_pydatetime(), "day", sampled_freq, region=REG_CN)
    np.testing.assert_array_equal(result, np.array([pd.Timestamp(value) for value in expected]))


@pytest.mark.parametrize("sampled_freq", ["week", "month", "2week", "2month"])
def test_resam_calendar_empty(sampled_freq):
    calendar = np.array([], dtype=object)
    np.testing.assert_array_equal(resam_calendar(calendar, "day", sampled_freq, region=REG_CN), calendar)
