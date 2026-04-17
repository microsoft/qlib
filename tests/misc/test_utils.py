from typing import List
from unittest.case import TestCase
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from qlib import init
from qlib.config import C
from qlib.log import TimeInspector
from qlib.constant import REG_CN, REG_US, REG_TW, REG_GB
from qlib.utils.time import (
    cal_sam_minute as cal_sam_minute_new,
    get_min_cal,
    is_single_value,
    time_to_day_index,
    CN_TIME,
    US_TIME,
    TW_TIME,
    GB_TIME,
)
from qlib.utils.data import guess_horizon

REG_MAP = {REG_CN: CN_TIME, REG_US: US_TIME, REG_TW: TW_TIME, REG_GB: GB_TIME}


def cal_sam_minute(x: pd.Timestamp, sam_minutes: int, region: str):
    """
    Sample raw calendar into calendar with sam_minutes freq, shift represents the shift minute the market time
        - open time of stock market is [9:30 - shift*pd.Timedelta(minutes=1)]
        - mid close time of stock market is [11:29 - shift*pd.Timedelta(minutes=1)]
        - mid open time of stock market is [13:00 - shift*pd.Timedelta(minutes=1)]
        - close time of stock market is [14:59 - shift*pd.Timedelta(minutes=1)]
    """
    # TODO: actually, this version is much faster when no cache or optimization
    day_time = pd.Timestamp(x.date())
    shift = C.min_data_shift
    region_time = REG_MAP[region]

    open_time = (
        day_time
        + pd.Timedelta(hours=region_time[0].hour, minutes=region_time[0].minute)
        - shift * pd.Timedelta(minutes=1)
    )
    close_time = (
        day_time
        + pd.Timedelta(hours=region_time[-1].hour, minutes=region_time[-1].minute)
        - shift * pd.Timedelta(minutes=1)
    )
    if region_time == CN_TIME:
        mid_close_time = (
            day_time
            + pd.Timedelta(hours=region_time[1].hour, minutes=region_time[1].minute - 1)
            - shift * pd.Timedelta(minutes=1)
        )
        mid_open_time = (
            day_time
            + pd.Timedelta(hours=region_time[2].hour, minutes=region_time[2].minute)
            - shift * pd.Timedelta(minutes=1)
        )
    else:
        mid_close_time = close_time
        mid_open_time = open_time

    if open_time <= x <= mid_close_time:
        minute_index = (x - open_time).seconds // 60
    elif mid_open_time <= x <= close_time:
        minute_index = (x - mid_open_time).seconds // 60 + 120
    else:
        raise ValueError("datetime of calendar is out of range")

    minute_index = minute_index // sam_minutes * sam_minutes

    if 0 <= minute_index < 120 or region_time != CN_TIME:
        return open_time + minute_index * pd.Timedelta(minutes=1)
    elif 120 <= minute_index < 240:
        return mid_open_time + (minute_index - 120) * pd.Timedelta(minutes=1)
    else:
        raise ValueError("calendar minute_index error, check `min_data_shift` in qlib.config.C")


class TimeUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        init()

    def test_cal_sam_minute(self):
        # test the correctness of the code
        random_n = 1000
        regions = [REG_CN, REG_US, REG_TW, REG_GB]

        def gen_args(cal: List):
            for time in np.random.choice(cal, size=random_n, replace=True):
                sam_minutes = np.random.choice([1, 2, 3, 4, 5, 6])
                dt = pd.Timestamp(
                    datetime(
                        2021,
                        month=3,
                        day=3,
                        hour=time.hour,
                        minute=time.minute,
                        second=time.second,
                        microsecond=time.microsecond,
                    )
                )
                args = dt, sam_minutes
                yield args

        for region in regions:
            cal_time = get_min_cal(region=region)
            for args in gen_args(cal_time):
                assert cal_sam_minute(*args, region) == cal_sam_minute_new(*args, region=region)

            # test the performance of the code
            args_l = list(gen_args(cal_time))

            with TimeInspector.logt():
                for args in args_l:
                    cal_sam_minute(*args, region=region)

            with TimeInspector.logt():
                for args in args_l:
                    cal_sam_minute_new(*args, region=region)


class GBTimeUtils(TestCase):
    """Tests for GB (London Stock Exchange) region support in time utils."""

    def test_get_min_cal_gb_count(self):
        # LSE trades 08:00–16:29 inclusive = 510 one-minute bars
        cal = get_min_cal(region=REG_GB)
        self.assertEqual(len(cal), 510)

    def test_get_min_cal_gb_open(self):
        cal = get_min_cal(region=REG_GB)
        self.assertEqual(cal[0].hour, 8)
        self.assertEqual(cal[0].minute, 0)

    def test_get_min_cal_gb_close(self):
        cal = get_min_cal(region=REG_GB)
        self.assertEqual(cal[-1].hour, 16)
        self.assertEqual(cal[-1].minute, 29)

    def test_is_single_value_gb_freq_too_small(self):
        # window smaller than freq → single value
        start = pd.Timestamp("2024-01-02 10:00:00")
        end = pd.Timestamp("2024-01-02 10:00:00")
        freq = pd.Timedelta("1min")
        self.assertTrue(is_single_value(start, end, freq, region=REG_GB))

    def test_is_single_value_gb_last_bar(self):
        # 16:29 is the last bar of the day → single value
        start = pd.Timestamp("2024-01-02 16:29:00")
        end = pd.Timestamp("2024-01-02 16:30:00")
        freq = pd.Timedelta("1min")
        self.assertTrue(is_single_value(start, end, freq, region=REG_GB))

    def test_is_single_value_gb_mid_session(self):
        # mid-session bar spanning a full minute → not single value
        start = pd.Timestamp("2024-01-02 12:00:00")
        end = pd.Timestamp("2024-01-02 12:01:00")
        freq = pd.Timedelta("1min")
        self.assertFalse(is_single_value(start, end, freq, region=REG_GB))

    def test_time_to_day_index_gb_open(self):
        # 08:00 is index 0
        self.assertEqual(time_to_day_index("8:00", region=REG_GB), 0)

    def test_time_to_day_index_gb_mid(self):
        # 12:00 = 240 minutes after 08:00
        self.assertEqual(time_to_day_index("12:00", region=REG_GB), 240)

    def test_time_to_day_index_gb_last(self):
        # 16:29 = 509 minutes after 08:00
        self.assertEqual(time_to_day_index("16:29", region=REG_GB), 509)

    def test_time_to_day_index_gb_out_of_range(self):
        # outside trading hours should raise
        with self.assertRaises(ValueError):
            time_to_day_index("7:59", region=REG_GB)
        with self.assertRaises(ValueError):
            time_to_day_index("16:30", region=REG_GB)


class DataUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        init()

    def test_guess_horizon(self):
        label = ["Ref($close, -2) / Ref($close, -1) - 1"]
        result = guess_horizon(label)
        assert result == 2

        label = ["Ref($close, -5) / Ref($close, -1) - 1"]
        result = guess_horizon(label)
        assert result == 5

        label = ["Ref($close, -1) / Ref($close, -1) - 1"]
        result = guess_horizon(label)
        assert result == 1


if __name__ == "__main__":
    unittest.main()
