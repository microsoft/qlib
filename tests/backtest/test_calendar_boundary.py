# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests for TradeCalendarManager.get_step_time at the right calendar boundary.

``get_step_time`` forms each step's right endpoint by peeking the *next* calendar bar
(``self._calendar[calendar_index + 1]``).  On the final step, ``calendar_index`` equals
``end_index``; if ``end_time`` is the last bar of the (future) calendar, that peek indexes
out of bounds and the backtest dies with an opaque ``IndexError`` deep inside
``get_step_time`` (see ``qlib/backtest/utils.py``, and the reports in #2278 / #1063).

The invariants exercised here:
    1. the final step must not raise IndexError;
    2. non-boundary steps keep the original ``epsilon_change(calendar[i + 1])`` right endpoint;
    3. pre-boundary steps are byte-for-byte identical to the old peek behaviour;
    4. the fallback works for ``day`` and an intraday frequency (``minute``);
    5. ``shift`` (which moves the effective boundary) is handled;
    6. the ``trade_step=None`` / current-step path is handled;
    7. ``finished()`` stays correct so the loop neither gains nor loses a step.
"""

import unittest

import numpy as np
import pandas as pd

from qlib.data import D
from qlib.backtest.utils import TradeCalendarManager
from qlib.utils.time import epsilon_change
from qlib.tests import TestAutoData


def make_manager(calendar, freq, start=0, end=None):
    """Build a TradeCalendarManager around an explicit, synthetic calendar.

    ``get_step_time`` only reads ``_calendar``, ``start_index``, ``freq`` and the current
    ``trade_step``, so its boundary logic can be exercised directly for any frequency
    without needing a matching data provider on disk.  The object is constructed without
    running ``reset`` (which would hit ``Cal.calendar``) precisely so the test controls the
    calendar contents.
    """
    tcm = object.__new__(TradeCalendarManager)
    tcm.freq = freq
    tcm._calendar = np.array([pd.Timestamp(x) for x in calendar])
    tcm.start_index = start
    tcm.end_index = len(calendar) - 1 if end is None else end
    tcm.trade_len = tcm.end_index - tcm.start_index + 1
    tcm.trade_step = 0
    tcm.start_time = tcm._calendar[tcm.start_index]
    tcm.end_time = tcm._calendar[tcm.end_index]
    return tcm


class TradeCalendarBoundaryTest(TestAutoData):
    """End-to-end reproduction against the real (future) calendar of the test dataset."""

    def test_get_step_time_at_last_calendar_bar(self):
        """The final step must not overflow when end_time is the last calendar bar (#2278/#1063)."""
        cal = D.calendar(future=True, freq="day")
        last = pd.Timestamp(cal[-1])
        prev = pd.Timestamp(cal[-2])

        tcm = TradeCalendarManager(freq="day", start_time=prev, end_time=last)
        last_step = tcm.get_trade_len() - 1

        # Before the fix this raises: IndexError: index N is out of bounds for axis 0 with size N
        start, end = tcm.get_step_time(last_step)

        self.assertEqual(start, last)
        # Right endpoint is well-defined: epsilon before the end of the last bar's period.
        self.assertEqual(end, epsilon_change(last + pd.Timedelta(days=1)))

    def test_non_boundary_step_unchanged(self):
        """A step that is not at the boundary keeps the original peek-the-next-bar behaviour."""
        cal = D.calendar(future=True, freq="day")
        end = pd.Timestamp(cal[-3])
        start = pd.Timestamp(cal[-5])

        tcm = TradeCalendarManager(freq="day", start_time=start, end_time=end)
        last_step = tcm.get_trade_len() - 1
        _, right = tcm.get_step_time(last_step)

        next_bar = pd.Timestamp(cal[-2])
        self.assertEqual(right, epsilon_change(next_bar))


class TradeCalendarBoundaryUnitTest(unittest.TestCase):
    """Frequency / shift / current-step / loop invariants on controlled synthetic calendars."""

    def test_boundary_fallback_day(self):
        """day: the last step's right endpoint is the end of that day."""
        cal = pd.date_range("2021-01-04", periods=5, freq="D")
        tcm = make_manager(cal, "day")
        start, end = tcm.get_step_time(tcm.get_trade_len() - 1)
        self.assertEqual(start, cal[-1])
        self.assertEqual(end, epsilon_change(cal[-1] + pd.Timedelta(days=1)))
        # single-bar interval preserved: end - start < one freq unit
        self.assertLess(end - start, pd.Timedelta(days=1))

    def test_boundary_fallback_minute(self):
        """minute: the same off-by-one is fixed at intraday granularity (criterion #4)."""
        cal = pd.date_range("2021-01-04 09:30", periods=5, freq="min")
        tcm = make_manager(cal, "1min")
        start, end = tcm.get_step_time(tcm.get_trade_len() - 1)
        self.assertEqual(start, cal[-1])
        self.assertEqual(end, epsilon_change(cal[-1] + pd.Timedelta(minutes=1)))
        self.assertLess(end - start, pd.Timedelta(minutes=1))

    def test_shift_moves_effective_boundary(self):
        """shift changes calendar_index, hence whether the boundary fallback applies (criterion #5)."""
        cal = pd.date_range("2021-01-04", periods=5, freq="D")
        tcm = make_manager(cal, "day")
        last_step = tcm.get_trade_len() - 1

        # shift=0: calendar_index == end_index (last bar) -> fallback.
        _, end0 = tcm.get_step_time(last_step, shift=0)
        self.assertEqual(end0, epsilon_change(cal[-1] + pd.Timedelta(days=1)))

        # shift=1: calendar_index == end_index - 1, so calendar[index + 1] still exists
        # -> original peek path, right endpoint is epsilon before the last real bar.
        start1, end1 = tcm.get_step_time(last_step, shift=1)
        self.assertEqual(start1, cal[-2])
        self.assertEqual(end1, epsilon_change(cal[-1]))

    def test_current_step_path(self):
        """trade_step=None uses the current step and must survive the boundary (criterion #6)."""
        cal = pd.date_range("2021-01-04", periods=5, freq="D")
        tcm = make_manager(cal, "day")
        tcm.trade_step = tcm.get_trade_len() - 1  # advance to the final step
        start, end = tcm.get_step_time()  # None -> get_trade_step()
        self.assertEqual(start, cal[-1])
        self.assertEqual(end, epsilon_change(cal[-1] + pd.Timedelta(days=1)))

    def test_full_loop_finished_and_step_count(self):
        """A full loop (as in #1063) yields exactly trade_len steps and finishes cleanly (criterion #7)."""
        cal = pd.date_range("2021-01-04", periods=5, freq="D")
        tcm = make_manager(cal, "day")

        steps = 0
        while not tcm.finished():
            tcm.get_step_time()  # must not raise, even on the final step
            tcm.step()
            steps += 1

        self.assertEqual(steps, tcm.get_trade_len())
        self.assertTrue(tcm.finished())

    def test_pre_boundary_steps_identical_to_old_peek(self):
        """Every pre-boundary step equals the original (calendar[i], epsilon(calendar[i+1])) (criterion #3)."""
        cal = pd.date_range("2021-01-04", periods=10, freq="D")
        tcm = make_manager(cal, "day", start=2, end=6)  # ends well before the calendar boundary
        for i in range(tcm.get_trade_len()):
            start, end = tcm.get_step_time(i)
            ci = tcm.start_index + i
            self.assertEqual(start, cal[ci])
            self.assertEqual(end, epsilon_change(cal[ci + 1]))


if __name__ == "__main__":
    unittest.main()
