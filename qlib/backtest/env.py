

import re
import json
import copy
import pathlib
import pandas as pd
from loguru import Logger
from ...data import D
from ...utils import get_date_in_file_name
from ...utils import get_pre_trading_date
from ..backtest.order import Order
from ..utils import init_instance_by_config

class BaseEnv:
    """
    # Strategy framework document

    class Env(BaseEnv):
    """

    def __init__(
        self,
        step_bar,
        trade_account,
        start_time=None,
        end_time=None,
        track=False,
        verbose=False,
        **kwargs
    ):
        self.step_bar = step_bar
        self.reset(start_time=start_time, end_time=end_time, trade_account=trade_account, track=track, **kwargs)

    def _reset_trade_date(self, start_time=None, end_time=None):
        if start_time:
            self.start_time = start_time
        if end_time:
            self.end_time = end_time
        if not self.start_time or not self.end_time:
            raise ValueError("value of `start_time` or `end_time` is None")
        _calendar = get_sample_freq_calendar(start_time=start_time, end_time=end_time, freq=step_bar)
        self.trade_dates = np.hstack(_calendar, pd.Timestamp(self.end_time))
        self.trade_len = len(self.trade_dates)
        self.trade_index = 0

    def reset(self, start_time=None, end_time=None, **kwargs):
        if start_time or end_time:
            self._reset_trade_date(start_time=start_time, end_time=end_time)
        self.track = kwargs.get("track", False)
        self.upper_action = kwargs.get("upper_action", None)
        self.trade_account = init_instance_by_config(kwargs.get("trade_account"))
        return self.trade_account

    def execute(self, **kwargs):
        self.trade_index = self.trade_index + 1
        return 
        (
            self.trade_account, 
            {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "trade_len": self.trade_len,
                "trade_index": self.trade_index - 1,
            }
        )

    def finished(self):
        return self.trade_index >= self.trade_len - 1



class SplitEnv(BaseEnv):
    def __init__(
        self, 
        step_bar, 
        start_time, 
        end_time, 
        trade_account,
        sub_env,
        sub_strategy,
        track=False, 
        verbose=False,
        **kwargs
    ):
        self.sub_env = sub_env
        self.sub_strategy = sub_strategy
        super(SplitEnv, self).__init__(step_bar=step_bar, start_time=start_time, end_time=end_time, trade_account=trade_account, track=track)
    
    def execute(self, order_list, **kwargs):
        if self.finished():
            raise StopIteration(f"this env has completed its task, please reset it if you want to call it!")
        #if self.track:
        #    yield action
        #episode_reward = 0
        trade_start_time = self.trade_dates[self.trade_index]
        trade_end_time = self.trade_dates[self.trade_index + 1]
        self.sub_strategy.reset(trade_order_list=order_list)
        sub_account = self.sub_env.reset(trade_order_list=order_list, start_time=self.trade_dates[self.trade_index - 1], end_time=self.trade_dates[self.trade_index])
        while not self.sub_env.finished():
            sub_order_list = self.sub_strategy.generate_order(sub_account)
            sub_account, sub_info = self.sub_env.execute(sub_order_list)
            #episode_reward += sub_reward
        _account, _info = super(SimulatorEnv, self).execute(**kwargs)
        return _account, _info


        
class SimulatorEnv(BaseEnv):

    def __init__(
        self, 
        step_bar, 
        start_time, 
        end_time, 
        trade_account, 
        trade_exchange,
        track=False, 
        verbose=False,
        **kwargs
    ):
        self.trade_exchange = trade_exchange
        super(SimulatorEnv, self).__init__(step_bar=step_bar, start_time=start_time, end_time=end_time, trade_account=trade_account, track=track, verbose=verbose)

    def execute(self, order_list, **kwargs):
        """
            Return: obs, done, info
        """
        if self.finished():
            raise StopIteration(f"this env has completed its task, please reset it if you want to call it!")

        trade_start_time = self.trade_dates[self.trade_index]
        trade_end_time = self.trade_dates[self.trade_index + 1]
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
        _account, _info = super(SimulatorEnv, self).execute(**kwargs)
        return _account, {**_info, "trade_info", trade_info}