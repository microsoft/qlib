# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
import bisect
from qlib.utils.time import epsilon_change
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from qlib.backtest.decision import BaseTradeDecision

import pandas as pd
import warnings

from ..data.data import Cal


class TradeCalendarManager:
    """
    Manager for trading calendar
        - BaseStrategy and BaseExecutor will use it
    """

    def __init__(
        self,
        freq: str,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        level_infra: "LevelInfrastructure" = None,
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
        self.level_infra = level_infra
        self.reset(freq=freq, start_time=start_time, end_time=end_time)

    def reset(self, freq, start_time, end_time):
        """
        Please refer to the docs of `__init__`

        Reset the trade calendar
        - self.trade_len : The total count for trading step
        - self.trade_step : The number of trading step finished, self.trade_step can be [0, 1, 2, ..., self.trade_len - 1]
        """
        self.freq = freq
        self.start_time = pd.Timestamp(start_time) if start_time else None
        self.end_time = pd.Timestamp(end_time) if end_time else None

        _calendar = Cal.calendar(freq=freq, future=True)
        self._calendar = _calendar
        _, _, _start_index, _end_index = Cal.locate_index(start_time, end_time, freq=freq, future=True)
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
        """get the total step length"""
        return self.trade_len

    def get_trade_step(self):
        return self.trade_step

    def get_step_time(self, trade_step=None, shift=0):
        """
        Get the left and right endpoints of the trade_step'th trading interval

        About the endpoints:
            - Qlib uses the closed interval in time-series data selection, which has the same performance as pandas.Series.loc
            # - The returned right endpoints should minus 1 seconds because of the closed interval representation in Qlib.
            # Note: Qlib supports up to minutely decision execution, so 1 seconds is less than any trading time interval.

        Parameters
        ----------
        trade_step : int, optional
            the number of trading step finished, by default None to indicate current step
        shift : int, optional
            shift bars , by default 0

        Returns
        -------
        Tuple[pd.Timestamp, pd.Timestap]
            - If shift == 0, return the trading time range
            - If shift > 0, return the trading time range of the earlier shift bars
            - If shift < 0, return the trading time range of the later shift bar
        """
        if trade_step is None:
            trade_step = self.get_trade_step()
        trade_step = trade_step - shift
        calendar_index = self.start_index + trade_step
        return self._calendar[calendar_index], epsilon_change(self._calendar[calendar_index + 1])

    def get_data_cal_range(self, rtype: str = "full") -> Tuple[int, int]:
        """
        get the calendar range
        The following assumptions are made
        1) The frequency of the exchange in common_infra is the same as the data calendar
        2) Users want the **data index** mod by **day** (i.e. 240 min)

        Parameters
        ----------
        rtype: str
            - "full": return the full limitation of the deicsion in the day
            - "step": return the limitation of current step

        Returns
        -------
        Tuple[int, int]:
        """
        # potential performance issue
        day_start = pd.Timestamp(self.start_time.date())
        day_end = epsilon_change(day_start + pd.Timedelta(days=1))
        freq = self.level_infra.get("common_infra").get("trade_exchange").freq
        _, _, day_start_idx, _ = Cal.locate_index(day_start, day_end, freq=freq)

        if rtype == "full":
            _, _, start_idx, end_index = Cal.locate_index(self.start_time, self.end_time, freq=freq)
        elif rtype == "step":
            _, _, start_idx, end_index = Cal.locate_index(*self.get_step_time(), freq=freq)
        else:
            raise ValueError(f"This type of input {rtype} is not supported")

        return start_idx - day_start_idx, end_index - day_start_idx

    def get_all_time(self):
        """Get the start_time and end_time for trading"""
        return self.start_time, self.end_time

    # helper functions
    def get_range_idx(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[int, int]:
        """
        get the range index which involve start_time~end_time  (both sides are closed)

        Parameters
        ----------
        start_time : pd.Timestamp
        end_time : pd.Timestamp

        Returns
        -------
        Tuple[int, int]:
            the index of the range.  **the left and right are closed**
        """
        left, right = (
            bisect.bisect_right(self._calendar, start_time) - 1,
            bisect.bisect_right(self._calendar, end_time) - 1,
        )
        left -= self.start_index
        right -= self.start_index

        def clip(idx):
            return min(max(0, idx), self.trade_len - 1)

        return clip(left), clip(right)

    def __repr__(self) -> str:
        return f"class: {self.__class__.__name__}; {self.start_time}[{self.start_index}]~{self.end_time}[{self.end_index}]: [{self.trade_step}/{self.trade_len}]"


class BaseInfrastructure:
    def __init__(self, **kwargs):
        self.reset_infra(**kwargs)

    def get_support_infra(self):
        raise NotImplementedError("`get_support_infra` is not implemented!")

    def reset_infra(self, **kwargs):
        support_infra = self.get_support_infra()
        for k, v in kwargs.items():
            if k in support_infra:
                setattr(self, k, v)
            else:
                warnings.warn(f"{k} is ignored in `reset_infra`!")

    def get(self, infra_name):
        if hasattr(self, infra_name):
            return getattr(self, infra_name)
        else:
            warnings.warn(f"infra {infra_name} is not found!")

    def has(self, infra_name):
        return infra_name in self.get_support_infra() and hasattr(self, infra_name)

    def update(self, other):
        support_infra = other.get_support_infra()
        infra_dict = {_infra: getattr(other, _infra) for _infra in support_infra if hasattr(other, _infra)}
        self.reset_infra(**infra_dict)


class CommonInfrastructure(BaseInfrastructure):
    def get_support_infra(self):
        return ["trade_account", "trade_exchange"]


class LevelInfrastructure(BaseInfrastructure):
    """level infrastructure is created by executor, and then shared to strategies on the same level"""

    def get_support_infra(self):
        """
        Descriptions about the infrastructure

        sub_level_infra:
        - **NOTE**: this will only work after _init_sub_trading !!!
        """
        return ["trade_calendar", "sub_level_infra", "common_infra"]

    def reset_cal(self, freq, start_time, end_time):
        """reset trade calendar manager"""
        if self.has("trade_calendar"):
            self.get("trade_calendar").reset(freq, start_time=start_time, end_time=end_time)
        else:
            self.reset_infra(
                trade_calendar=TradeCalendarManager(freq, start_time=start_time, end_time=end_time, level_infra=self)
            )

    def set_sub_level_infra(self, sub_level_infra: LevelInfrastructure):
        """this will make the calendar access easier when acrossing multi-levels"""
        self.reset_infra(sub_level_infra=sub_level_infra)


def get_start_end_idx(trade_calendar: TradeCalendarManager, outer_trade_decision: BaseTradeDecision) -> Union[int, int]:
    """
    A helper function for getting the decision-level index range limitation for inner strategy
    - NOTE: this function is not applicable to order-level

    Parameters
    ----------
    trade_calendar : TradeCalendarManager
    outer_trade_decision : BaseTradeDecision
        the trade decision made by outer strategy

    Returns
    -------
    Union[int, int]:
        start index and end index
    """
    try:
        return outer_trade_decision.get_range_limit(inner_calendar=trade_calendar)
    except NotImplementedError:
        return 0, trade_calendar.get_trade_len() - 1
