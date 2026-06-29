import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd


class TradeCalendarManagerTest(unittest.TestCase):
    @staticmethod
    def _load_trade_calendar_manager():
        utils_path = Path(__file__).resolve().parents[2] / "qlib" / "backtest" / "utils.py"

        qlib_pkg = ModuleType("qlib")
        qlib_pkg.__path__ = []
        backtest_pkg = ModuleType("qlib.backtest")
        backtest_pkg.__path__ = []
        data_pkg = ModuleType("qlib.data")
        data_pkg.__path__ = []
        utils_pkg = ModuleType("qlib.utils")
        utils_pkg.__path__ = []

        time_module = ModuleType("qlib.utils.time")
        time_module.epsilon_change = lambda dt, direction="backward": dt - pd.Timedelta(seconds=1)
        class FakeFreq:
            @staticmethod
            def parse(freq: str):
                if freq == "day":
                    return 1, "day"
                raise ValueError(freq)

            @staticmethod
            def get_timedelta(n: int, freq: str):
                return pd.Timedelta(f"{n}{freq}")

        time_module.Freq = FakeFreq

        data_module = ModuleType("qlib.data.data")

        class FakeCal:
            @staticmethod
            def calendar(*args, **kwargs):
                raise NotImplementedError

            @staticmethod
            def locate_index(*args, **kwargs):
                raise NotImplementedError

        data_module.Cal = FakeCal

        with patch.dict(
            "sys.modules",
            {
                "qlib": qlib_pkg,
                "qlib.backtest": backtest_pkg,
                "qlib.data": data_pkg,
                "qlib.data.data": data_module,
                "qlib.utils": utils_pkg,
                "qlib.utils.time": time_module,
            },
        ):
            spec = spec_from_file_location("qlib.backtest.utils", utils_path)
            module = module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module.TradeCalendarManager, module.Cal

    def test_get_step_time_uses_freq_boundary_when_future_calendar_is_missing(self):
        TradeCalendarManager, Cal = self._load_trade_calendar_manager()
        calendar = np.array(
            [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-04"),
            ],
            dtype=object,
        )

        with patch.object(Cal, "calendar", return_value=calendar), patch.object(
            Cal,
            "locate_index",
            return_value=(calendar[0], calendar[-1], 0, 2),
        ):
            trade_calendar = TradeCalendarManager(freq="day", start_time="2024-01-02", end_time="2024-01-04")

        step_start, step_end = trade_calendar.get_step_time(trade_step=2)

        self.assertEqual(pd.Timestamp("2024-01-04"), step_start)
        self.assertEqual(pd.Timestamp("2024-01-04 23:59:59"), step_end)

    def test_get_step_time_keeps_next_calendar_boundary_when_available(self):
        TradeCalendarManager, Cal = self._load_trade_calendar_manager()
        calendar = np.array(
            [
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-08"),
                pd.Timestamp("2024-01-09"),
            ],
            dtype=object,
        )

        with patch.object(Cal, "calendar", return_value=calendar), patch.object(
            Cal,
            "locate_index",
            return_value=(calendar[0], calendar[-1], 0, 2),
        ):
            trade_calendar = TradeCalendarManager(freq="day", start_time="2024-01-05", end_time="2024-01-09")

        step_start, step_end = trade_calendar.get_step_time(trade_step=0)

        self.assertEqual(pd.Timestamp("2024-01-05"), step_start)
        self.assertEqual(pd.Timestamp("2024-01-07 23:59:59"), step_end)


if __name__ == "__main__":
    unittest.main()
