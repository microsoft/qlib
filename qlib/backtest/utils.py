# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from typing import Union

from ..utils.resam import get_resam_calendar
from ..data.data import Cal


class TradeCalendarManager:
    """
    Manager for trading calendar
        - BaseStrategy and BaseExecutor will use it
    """

    def __init__(
        self, freq: str, start_time: Union[str, pd.Timestamp] = None, end_time: Union[str, pd.Timestamp] = None
    ):
        """
        Parameters
        ----------
        freq : str
            frequency of trading calendar, also trade time per trading step
        start_time : Union[str, pd.Timestamp], optional
            closed start of the trading calendar, by default None
            If `start_time` is None, it must be reset before trading.
        end_time : Union[str, pd.Timestamp], optional
            closed end of the trade time range, by default None
            If `end_time` is None, it must be reset before trading.
        """
        self.freq = freq
        self.start_time = pd.Timestamp(start_time) if start_time else None
        self.end_time = pd.Timestamp(end_time) if end_time else None
        self._init_trade_calendar(freq=freq, start_time=start_time, end_time=end_time)

    def _init_trade_calendar(self, freq, start_time, end_time):
        """
        Reset the trade calendar
        - self.trade_len : The total count for trading step
        - self.trade_step : The number of trading step finished, self.trade_step can be [0, 1, 2, ..., self.trade_len - 1]
        """
        _calendar, freq, freq_sam = get_resam_calendar(freq=freq)
        self.trade_calendar = _calendar
        _, _, _start_index, _end_index = Cal.locate_index(start_time, end_time, freq=freq, freq_sam=freq_sam)
        self.start_index = _start_index
        self.end_index = _end_index
        self.trade_len = _end_index - _start_index + 1
        self.trade_step = 0

    def finished(self):
        """
        Check if the trading finished
        - Should check before calling strategy.generate_decisions and executor.execute
        - If self.trade_step >= self.self.trade_len, it means the trading is finished
        - If self.trade_step < self.self.trade_len, it means the number of trading step finished is self.trade_step
        """
        return self.trade_step >= self.trade_len

    def step(self):
        if self.finished():
            raise RuntimeError(f"The calendar is finished, please reset it if you want to call it!")
        self.trade_step = self.trade_step + 1

    def get_freq(self):
        return self.freq

    def get_trade_len(self):
        return self.trade_len

    def get_trade_step(self):
        return self.trade_step

    def get_step_time(self, trade_step=0, shift=0):
        """
        Get the time range of trading step

        Parameters
        ----------
        trade_step : int, optional
            the number of trading step finished, by default 0
        shift : int, optional
            shift bars , by default 0

        Returns
        -------
        Tuple[pd.Timestamp, pd.Timestap]
            - If shift == 0, return the trading time range
            - If shift > 0, return the trading time range of the earlier shift bars
            - If shift < 0, return the trading time range of the later shift bar
        """
        trade_step = trade_step - shift
        calendar_index = self.start_index + trade_step
        return self.trade_calendar[calendar_index], self.trade_calendar[calendar_index + 1] - pd.Timedelta(seconds=1)

    def get_all_time(self):
        """Get the start_time and end_time for trading"""
        return self.start_time, self.end_time
