

import re
import json
import copy
import warnings
import pathlib
import numpy as np
import pandas as pd
from ...data.data import Cal
from ...utils import get_sample_freq_calendar
from .order import Order


class TradeCalendarBase:

    def _reset_trade_calendar(self, start_time, end_time):
        if start_time:
            self.start_time = pd.Timestamp(start_time)
        if end_time:
            self.end_time = pd.Timestamp(end_time)
        if self.start_time and self.end_time:
            _calendar, freq, freq_sam = get_sample_freq_calendar(freq=self.step_bar)
            self.calendar = _calendar
            _start_time, _end_time, _start_index, _end_index = Cal.locate_index(self.start_time, self.end_time, freq=freq, freq_sam=freq_sam)
            _trade_calendar = self.calendar[_start_index: _end_index + 1]
            if _start_time != self.start_time:
                self.trade_calendar = np.hstack((self.start_time, _trade_calendar, self.end_time))
                self.start_index = _start_index - 1
            else:
                self.trade_calendar = np.hstack((_trade_calendar, self.end_time))
                self.start_index = _start_index
            self.end_index = _end_index
            self.trade_index = 0
            self.trade_len = len(self.trade_calendar)
        else:
            raise ValueError("failed to reset trade calendar, params `start_time` or `end_time` is None.")

    def _get_trade_time(self, trade_index=1, shift=0):
        trade_index = trade_index - shift
        if 0 < trade_index < self.trade_len - 1: 
            trade_start_time = self.trade_calendar[trade_index - 1]
            trade_end_time = self.trade_calendar[trade_index] - pd.Timedelta(seconds=1)
            return trade_start_time, trade_end_time
        elif trade_index == self.trade_len - 1:
            trade_start_time = self.trade_calendar[trade_index - 1]
            trade_end_time = self.trade_calendar[trade_index]
            return trade_start_time, trade_end_time
        else:
            raise RuntimeError("trade_index out of range")
    
    def _get_calendar_time(self, trade_index=1, shift=1):
        trade_index = trade_index - shift
        calendar_index = self.start_index + trade_index
        return self.calendar[calendar_index - 1], self.calendar[calendar_index]

class BaseEnv(TradeCalendarBase):
    """
    # Strategy framework document

    class Env(BaseEnv):
    """

    def __init__(
        self,
        step_bar,
        start_time=None,
        end_time=None,
        trade_account=None,
        verbose=False,
        **kwargs,
    ):
        self.step_bar = step_bar
        self.verbose = verbose
        self.reset(start_time=start_time, end_time=end_time, trade_account=trade_account, **kwargs)

    def _get_position(self):
        return self.trade_account.current
    

    def reset(self, start_time=None, end_time=None, trade_account=None, **kwargs):
        if start_time or end_time:
            self._reset_trade_calendar(start_time=start_time, end_time=end_time)
        if trade_account:
            self.trade_account = trade_account
        
        for k, v in kwargs:
            if hasattr(self, k):
                setattr(self, k, v)

    def get_init_state(self):
        init_state = {"current": self._get_position()}
        return init_state
    

    def execute(self, order_list=None, **kwargs):
        self.trade_index = self.trade_index + 1

    def finished(self):
        return self.trade_index >= self.trade_len - 1


class SplitEnv(BaseEnv):
    def __init__(
        self, 
        step_bar, 
        sub_env,
        sub_strategy,
        start_time=None, 
        end_time=None, 
        trade_account=None,
        verbose=False,
        **kwargs
    ):
        self.sub_env = sub_env
        self.sub_strategy = sub_strategy
        super(SplitEnv, self).__init__(step_bar=step_bar, start_time=start_time, end_time=end_time, trade_account=trade_account, verbose=verbose)
    
    def execute(self, order_list, **kwargs):
        if self.finished():
            raise StopIteration(f"this env has completed its task, please reset it if you want to call it!")
        #if self.track:
        #    yield action
        #episode_reward = 0
        super(SplitEnv, self).execute(**kwargs)
        trade_start_time, trade_end_time = self._get_trade_time(trade_index=self.trade_index)
        self.sub_env.reset(start_time=trade_start_time, end_time=trade_end_time, trade_account=self.trade_account)
        self.sub_strategy.reset(start_time=trade_start_time, end_time=trade_end_time, trade_order_list=order_list)
        trade_state = self.sub_env.get_init_state()
        while not self.sub_env.finished():
            _order_list = self.sub_strategy.generate_order_list(**trade_state)
            trade_state, trade_info = self.sub_env.execute(order_list=_order_list)
            #episode_reward += sub_reward
        _obs = {"current": self._get_position()}
        _info = {}
        return _obs, _info


        
class SimulatorEnv(BaseEnv):

    def __init__(
        self, 
        step_bar, 
        start_time=None, 
        end_time=None, 
        trade_account=None, 
        trade_exchange=None,
        verbose=False,
        **kwargs,
    ):
        super(SimulatorEnv, self).__init__(step_bar=step_bar, start_time=start_time, end_time=end_time, trade_account=trade_account, trade_exchange=trade_exchange, verbose=verbose, **kwargs)

    def reset(self, trade_exchange=None, **kwargs):
        super(SimulatorEnv, self).reset(**kwargs)
        if trade_exchange:
            self.trade_exchange=trade_exchange

    def execute(self, order_list, **kwargs):
        """
            Return: obs, done, info
        """
        if self.finished():
            raise StopIteration(f"this env has completed its task, please reset it if you want to call it!")
        super(SimulatorEnv, self).execute(**kwargs)
        trade_start_time, trade_end_time = self._get_trade_time(trade_index=self.trade_index)
        trade_info = []
        for order in order_list:
            if self.trade_exchange.check_order(order) is True:
                # execute the order
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(order, trade_account=self.trade_account)
                trade_info.append((order, trade_val, trade_cost, trade_price))
                if self.verbose:
                    if order.direction == Order.SELL:  # sell
                        print(
                            "[I {:%Y-%m-%d}]: sell {}, price {:.2f}, amount {}, value {:.2f}.".format(
                                trade_start_time,
                                order.stock_id,
                                trade_price,
                                order.deal_amount,
                                trade_val,
                            )
                        )
                    else:
                        print(
                            "[I {:%Y-%m-%d}]: buy {}, price {:.2f}, amount {}, value {:.2f}.".format(
                                trade_start_time,
                                order.stock_id,
                                trade_price,
                                order.deal_amount,
                                trade_val,
                            )
                        )

            else:
                if self.verbose:
                    print("[W {:%Y-%m-%d}]: {} wrong.".format(trade_start_time, order.stock_id))
                # do nothing
                pass
        self.trade_account.update_bar_end(trade_start_time=trade_start_time, trade_end_time=trade_end_time, trade_exchange=self.trade_exchange)
        _obs = {"current": self._get_position()}
        _info = {"trade_info": trade_info}
        return _obs, _info