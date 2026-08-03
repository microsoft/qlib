import numpy as np
import pandas as pd

from qlib.backtest.utils import TradeCalendarManager


def _make_calendar_manager():
    trade_calendar = TradeCalendarManager.__new__(TradeCalendarManager)
    trade_calendar.freq = "day"
    trade_calendar.start_index = 0
    trade_calendar.end_index = 1
    trade_calendar.trade_len = 2
    trade_calendar.trade_step = 1
    trade_calendar._calendar = np.array(
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    )
    return trade_calendar


def test_get_step_time_uses_next_calendar_when_available():
    trade_calendar = _make_calendar_manager()

    start_time, end_time = trade_calendar.get_step_time(trade_step=0)

    assert start_time == pd.Timestamp("2020-01-01")
    assert end_time == pd.Timestamp("2020-01-01 23:59:59")


def test_get_step_time_infers_right_boundary_for_last_calendar_bar():
    trade_calendar = _make_calendar_manager()

    start_time, end_time = trade_calendar.get_step_time()

    assert start_time == pd.Timestamp("2020-01-02")
    assert end_time == pd.Timestamp("2020-01-02 23:59:59")
