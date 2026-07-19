# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tests for the calendar right-boundary fix in TradeCalendarManager.get_step_time.

Before the fix, calling get_step_time on the very last trading step raised:
    IndexError: index N is out of bounds for axis 0 with size N
because it always tried to read calendar[index + 1] even when index was the
last position.

These tests are intentionally self-contained: they load only qlib/utils/time.py
and qlib/backtest/utils.py via importlib so the test can run without a full
qlib installation or qlib/__init__.py side-effects.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import the two modules we care about without triggering qlib/__init__.py
# ---------------------------------------------------------------------------

QLIB_ROOT = Path(__file__).resolve().parents[2] / "qlib"


def _load_module(rel_path: str, name: str):
    """Load a single .py file as a module, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, QLIB_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# qlib.utils.time has no heavy deps — load it directly
time_mod = _load_module("utils/time.py", "qlib.utils.time")
epsilon_change = time_mod.epsilon_change
Freq = time_mod.Freq

# backtest/utils.py imports from qlib.utils.time and qlib.data.data (Cal).
# We stub out qlib.data.data so only the parts we test are exercised.
_fake_qlib_data = types.ModuleType("qlib.data.data")
_fake_qlib_data.Cal = None  # Cal is not used by get_step_time
sys.modules.setdefault("qlib.data", types.ModuleType("qlib.data"))
sys.modules["qlib.data.data"] = _fake_qlib_data

# backtest/decision is only imported under TYPE_CHECKING — safe to stub
_fake_decision = types.ModuleType("qlib.backtest.decision")
_fake_decision.BaseTradeDecision = object
sys.modules.setdefault("qlib.backtest", types.ModuleType("qlib.backtest"))
sys.modules["qlib.backtest.decision"] = _fake_decision

utils_mod = _load_module("backtest/utils.py", "qlib.backtest.utils")
TradeCalendarManager = utils_mod.TradeCalendarManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(dates: list, start: str, end: str, freq: str = "day") -> TradeCalendarManager:
    """Build a TradeCalendarManager backed by a hand-crafted numpy calendar.

    qlib stores calendars as object-dtype numpy arrays of pd.Timestamp values,
    so we match that exactly here.
    """
    # object-dtype array of Timestamps — same as what Cal.calendar() returns
    cal = np.array([pd.Timestamp(d) for d in dates], dtype=object)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_idx = int(np.searchsorted(cal, start_ts))
    end_idx = int(np.searchsorted(cal, end_ts))

    mgr = object.__new__(TradeCalendarManager)
    mgr.freq = freq
    mgr.start_time = start_ts
    mgr.end_time = end_ts
    mgr._calendar = cal
    mgr.start_index = start_idx
    mgr.end_index = end_idx
    mgr.trade_len = end_idx - start_idx + 1
    mgr.trade_step = 0
    mgr.level_infra = None
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalendarBoundary(unittest.TestCase):
    # three trading days, no future extension
    DATES = [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]

    def _mgr(self, start="2024-01-02", end="2024-01-04", freq="day"):
        return _make_manager(self.DATES, start, end, freq=freq)

    def test_non_boundary_step_unchanged(self):
        """Steps that aren't at the calendar end keep the original epsilon_change(next_bar) logic."""
        mgr = self._mgr()
        left, right = mgr.get_step_time(trade_step=0)
        self.assertEqual(left, self.DATES[0])
        self.assertEqual(right, epsilon_change(self.DATES[1]))

    def test_boundary_step_does_not_raise(self):
        """get_step_time on the very last bar must not raise IndexError."""
        mgr = self._mgr()
        last_step = mgr.trade_len - 1  # calendar_index == len(calendar)-1
        try:
            mgr.get_step_time(trade_step=last_step)
        except IndexError as exc:
            self.fail(f"get_step_time raised IndexError on the last bar: {exc}")

    def test_boundary_fallback_day(self):
        """Fallback right = left + 1 day, epsilon_change applied."""
        mgr = self._mgr()
        last_step = mgr.trade_len - 1
        left, right = mgr.get_step_time(trade_step=last_step)
        expected_right = epsilon_change(self.DATES[2] + Freq.get_timedelta(*Freq.parse("day")))
        self.assertEqual(left, self.DATES[2])
        self.assertEqual(right, expected_right)

    def test_boundary_fallback_minute(self):
        """Same fallback logic works for minute-frequency calendars."""
        base = pd.Timestamp("2024-01-02 09:30")
        dates = [base + pd.Timedelta(minutes=i) for i in range(3)]
        mgr = _make_manager(dates, str(dates[0]), str(dates[2]), freq="1min")
        last_step = mgr.trade_len - 1
        left, right = mgr.get_step_time(trade_step=last_step)
        expected_right = epsilon_change(dates[2] + Freq.get_timedelta(*Freq.parse("1min")))
        self.assertEqual(left, dates[2])
        self.assertEqual(right, expected_right)

    def test_pre_boundary_steps_identical_to_old_peek(self):
        """Every step except the last returns the same result as the original calendar[i+1] peek."""
        mgr = self._mgr()
        cal = mgr._calendar
        for step in range(mgr.trade_len - 1):
            idx = mgr.start_index + step
            left, right = mgr.get_step_time(trade_step=step)
            self.assertEqual(left, cal[idx])
            self.assertEqual(right, epsilon_change(cal[idx + 1]))

    def test_full_loop_step_count(self):
        """Loop through all steps — exactly trade_len iterations, finished() True at end."""
        mgr = self._mgr()
        count = 0
        while not mgr.finished():
            mgr.get_step_time()  # must not raise
            mgr.step()
            count += 1
        self.assertEqual(count, mgr.trade_len)
        self.assertTrue(mgr.finished())


if __name__ == "__main__":
    unittest.main()
