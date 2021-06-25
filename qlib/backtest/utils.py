# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.backtest.order import Order
from qlib.strategy.base import BaseStrategy
from qlib.backtest.exchange import Exchange
from qlib.backtest.account import Account
import pandas as pd
import warnings
from typing import Tuple, Union, List, Set

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
        self._calendar = _calendar
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
        Get the left and right endpoints of the trade_step'th trading interval

        About the endpoints:
            - Qlib uses the closed interval in time-series data selection, which has the same performance as pandas.Series.loc
            - The returned right endpoints should minus 1 seconds becasue of the closed interval representation in Qlib.
            Note: Qlib supports up to minutely decision execution, so 1 seconds is less than any trading time interval.

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
        return self._calendar[calendar_index], self._calendar[calendar_index + 1] - pd.Timedelta(seconds=1)

    def get_all_time(self):
        """Get the start_time and end_time for trading"""
        return self.start_time, self.end_time


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
        if infra_name in self.get_support_infra() and hasattr(self, infra_name):
            return True
        else:
            return False

    def update(self, other):
        support_infra = other.get_support_infra()
        infra_dict = {_infra: getattr(other, _infra) for _infra in support_infra if hasattr(other, _infra)}
        self.reset_infra(**infra_dict)



class CommonInfrastructure(BaseInfrastructure):
    def get_support_infra(self):
        return ["trade_account", "trade_exchange"]


class LevelInfrastructure(BaseInfrastructure):
    def get_support_infra(self):
        return ["trade_calendar"]


class BaseTradeDecision:
    # TODO: put it into order.py; and replace it with decision.py
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy

    def get_decision(self) -> List[object]:
        """
        get the **concrete decision**  (e.g. concrete decision)
        This will be called by the inner strategy

        Returns
        -------
        List[object]:
            The decision result. Typically it is some orders
            Example:
                []:
                    Decision not available
                concrete_decision:
                    available
        """
        raise NotImplementedError(f"This type of input is not supported")

    def update(self, trade_calendar: TradeCalendarManager) -> "BaseTradeDecison":
        """
        Be called at the **start** of each step

        Parameters
        ----------
        trade_calendar : TradeCalendarManager
            The calendar of the **inner strategy**!!!!!

        Returns
        -------
        None:
            No update, use previous decision(or unavailable)
        BaseTradeDecison:
            New update, use new decision
        """
        return self.strategy.update_trade_decision(self, trade_calendar)

    def get_range_limit(self) -> Tuple[int, int]:
        """
        return the expected step range for limiting the decision execution time

        Returns
        -------
        Tuple[int, int]:

        Raises
        ------
        NotImplementedError:
            If the decision can't provide a unified start and end
        """
        raise NotImplementedError(f"Please implement the `func` method")


class TradeDecisonWO(BaseTradeDecision):
    def __init__(self, order_list: List[Order], strategy: BaseStrategy):
        super().__init__(strategy)
        self.order_list = order_list


class TradeDecison(BaseTradeDecision):
    """trade decision that made by strategy"""

    def __init__(self, order_list, ori_strategy, init_enable=False):
        """
        Parameters
        ----------
        order_list : list
            the order list
        ori_strategy : BaseStrategy
            the original strategy that make the decison
        init_enable : bool, optional
            wether to enable order initially, default by False
        """
        self.order_list = order_list
        self.ori_strategy = ori_strategy
        if init_enable:
            self.enable_dict = {_order.stock_id: _order for _order in self.order_list}
            self.disable_dict = dict()
        else:
            self.enable_dict = dict()
            self.disable_dict = {_order.stock_id: _order for _order in self.order_list}

    def enable(self, enable_set: Union[List[str], Set[str]] = None, all_enable=False):
        """enable order set
        Parameters
        ----------
        enable_set : Union[List[str], Set[str]], optional
            the order set that will be enabled, by default None
            - if all_enable is True, enable_set will be ignored
            - else, enable the order whose stock_id in enable_set
        all_enable : bool, optional
            wether to enable all order, by default False
        """
        if all_enable is True:
            self.enable_dict.update(self.disable_dict)
            self.disable_dict.clear()
            if enable_set is not None:
                warnings.warn(f"`enable_set` is ignored because `all_enable` is set True")
        else:
            enable_set = set(enable_set)
            for _stock_id in enable_set:
                enable_order = self.disable_dict.get(_stock_id)
                if enable_order is None:
                    raise ValueError(f"_stock_id {_stock_id} is not found in disable set")
                self.enable_order.update({_stock_id: enable_order})
                self.disable_dict.pop(_stock_id)

    def disable(self, disable_set: Union[List[str], Set[str]] = None, all_disable=False):
        """disable order set
        Parameters
        ----------
        disable_set : Union[List[str], Set[str]], optional
            the order set that will be disabled, by default None
            - if all_disable is True, disable_set will be ignored
            - else, disable the order whose stock_id in disable_set
        all_disable : bool, optional
            wether to disable all order, by default False
        """
        if all_disable is True:
            self.disable_dict.update(self.enable_dict)
            self.enable_dict.clear()
            if disable_set is not None:
                warnings.warn(f"`disable_set` is ignored because `all_disable` is set True")
        else:
            disable_set = set(disable_set)
            for _stock_id in disable_set:
                disable_order = self.enable_dict.get(_stock_id)
                if disable_order is None:
                    raise ValueError(f"_stock_id {_stock_id} is not found in enable set")
                self.disable_dict.update({_stock_id: disable_order})
                self.enable_dict.pop(_stock_id)

    def generator(self, only_enable=False, only_disable=False):
        """get order generator used for iteration
        Parameters
        ----------
        only_enable : bool, optional
            wether to ignore disabled order, by default False
        only_disable : bool, optional
            wether to ignore enabled order, by default False
        """
        if not only_disable and not only_enable:
            yield from self.order_list
        elif not only_disable:
            yield from self.enable_dict.values()
        elif not only_enable:
            yield from self.disable_dict.values()

    def get_order_list(self, only_enable=False, only_disable=False):
        """get the order list

        Parameters
        ----------
        only_enable : bool, optional
            wether to ignore disabled order, by default False
        only_disable : bool, optional
            wether to ignore enabled order, by default False
        Returns
        -------
        List[Order]
            the order list
        """
        if not only_disable and not only_enable:
            return self.order_list
        elif not only_disable:
            return list(self.enable_dict.values())
        elif not only_enable:
            return list(self.disable_dict.values())

    def update(self, trade_calendar: TradeCalendarManager):
        """
        make the original strategy update the enabled status of orders.

        Parameters
        ----------
        trade_calendar : TradeCalendarManager
            the trade calendar for sub strategy
        """
        self.ori_strategy.update_trade_decision(self, trade_calendar)
