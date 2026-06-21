# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression test for TradeCalendarManager.get_step_time at the right calendar boundary.

``get_step_time`` forms each step's right endpoint by peeking the *next* calendar bar
(``self._calendar[calendar_index + 1]``).  On the final step, ``calendar_index`` equals
``end_index``; if ``end_time`` is the last bar of the (future) calendar, that peek indexes
out of bounds and the backtest dies with an opaque ``IndexError`` deep inside
``get_step_time`` (see ``qlib/backtest/utils.py``).  A backtest that ends on the last
available calendar bar must instead produce a well-defined final interval.
"""

import unittest

import pandas as pd

from qlib.data import D
from qlib.backtest.utils import TradeCalendarManager
from qlib.tests import TestAutoData


class TradeCalendarBoundaryTest(TestAutoData):
    def test_get_step_time_at_last_calendar_bar(self):
        """The final step must not overflow when end_time is the last calendar bar."""
        cal = D.calendar(future=True, freq="day")
        last = pd.Timestamp(cal[-1])
        prev = pd.Timestamp(cal[-2])

        tcm = TradeCalendarManager(freq="day", start_time=prev, end_time=last)
        last_step = tcm.get_trade_len() - 1

        # Before the fix this raises: IndexError: index N is out of bounds for axis 0 with size N
        start, end = tcm.get_step_time(last_step)

        # Left endpoint is the last bar itself.
        self.assertEqual(start, last)
        # Right endpoint is well-defined and stays within the last bar's period (a single bar):
        # start < end < start + 1 day.
        self.assertGreater(end, start)
        self.assertLess(end, last + pd.Timedelta(days=1))

    def test_non_boundary_step_unchanged(self):
        """A step that is not at the boundary keeps the original peek-the-next-bar behaviour."""
        cal = D.calendar(future=True, freq="day")
        # End two bars before the calendar end so calendar[index + 1] still exists.
        end = pd.Timestamp(cal[-3])
        start = pd.Timestamp(cal[-5])

        tcm = TradeCalendarManager(freq="day", start_time=start, end_time=end)
        last_step = tcm.get_trade_len() - 1
        _, right = tcm.get_step_time(last_step)

        # Original behaviour: right endpoint is epsilon before the next real calendar bar.
        next_bar = pd.Timestamp(cal[-2])
        self.assertEqual(right, next_bar - pd.Timedelta(seconds=1))


if __name__ == "__main__":
    unittest.main()
