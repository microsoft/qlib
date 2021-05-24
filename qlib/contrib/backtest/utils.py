# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from typing import Union

from ...utils.resam import get_resam_calendar
from ...data.data import Cal


class TradeCalendarManager:
    """
    Manager for trading calendar
        - BaseStrategy and BaseExecutor will use it
    """

    def __init__(
        self, step_bar: str, start_time: Union[str, pd.Timestamp] = None, end_time: Union[str, pd.Timestamp] = None
    ):
        """
        Parameters
        ----------
        step_bar : str
            frequency of each trading calendar
        start_time : Union[str, pd.Timestamp], optional
            closed start of the trading calendar, by default None
            If `start_time` is None, it must be reset before trading.
        end_time : Union[str, pd.Timestamp], optional
            closed end of the trade time range, by default None
            If `end_time` is None, it must be reset before trading.
        """
        self.step_bar = step_bar
        self.start_time = pd.Timestamp(start_time) if start_time else None
        self.end_time = pd.Timestamp(start_time) if start_time else None
        self._init_trade_calendar(step_bar=step_bar, start_time=start_time, end_time=end_time)

    def _init_trade_calendar(self, step_bar, start_time, end_time):
        """reset trade calendar"""
        _calendar, freq, freq_sam = get_resam_calendar(freq=step_bar)
        self.calendar = _calendar
        _, _, _start_index, _end_index = Cal.locate_index(start_time, end_time, freq=freq, freq_sam=freq_sam)
        self.start_index = _start_index
        self.end_index = _end_index
        self.trade_len = _end_index - _start_index + 1
        self.trade_index = 0

    def finished(self):
        return self.trade_index >= self.trade_len

    def step(self):
        if self.finished():
            raise RuntimeError(f"The calendar is finished, please reset it if you want to call it!")
        self.trade_index = self.trade_index + 1

    def get_step_bar(self):
        return self.step_bar

    def get_trade_len(self):
        return self.trade_len

    def get_trade_index(self):
        return self.trade_index

    def get_calendar_time(self, trade_index=1, shift=0):
        trade_index = trade_index - shift
        calendar_index = self.start_index + trade_index
        return self.calendar[calendar_index - 1], self.calendar[calendar_index] - pd.Timedelta(seconds=1)
