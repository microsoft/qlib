import re
import json
import copy
import warnings
import pathlib
import numpy as np
import pandas as pd
from ...data.data import Cal
from ...utils import get_sample_freq_calendar, parse_freq
from .position import Position
from .report import Report
from .order import Order


class BaseTradeCalendar:
    def __init__(self, step_bar, start_time=None, end_time=None, **kwargs):
        self.step_bar = step_bar
        self.reset(start_time=start_time, end_time=end_time)

    def _reset_trade_calendar(self, start_time, end_time):
        if not start_time and not end_time:
            return
        if start_time:
            self.start_time = pd.Timestamp(start_time)
        if end_time:
            self.end_time = pd.Timestamp(end_time)
        if self.start_time and self.end_time:
            _calendar, freq, freq_sam = get_sample_freq_calendar(freq=self.step_bar)
            self.calendar = _calendar
            _start_time, _end_time, _start_index, _end_index = Cal.locate_index(
                self.start_time, self.end_time, freq=freq, freq_sam=freq_sam
            )
            _trade_calendar = self.calendar[_start_index : _end_index + 1]
            self.start_index = _start_index
            self.end_index = _end_index
            self.trade_len = _end_index - _start_index + 1
            self.trade_index = 0
        else:
            raise ValueError("failed to reset trade calendar, params `start_time` or `end_time` is None.")

    def reset(self, start_time=None, end_time=None, **kwargs):
        if start_time or end_time:
            self._reset_trade_calendar(start_time=start_time, end_time=end_time)

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _get_calendar_time(self, trade_index=1, shift=0):
        trade_index = trade_index - shift
        calendar_index = self.start_index + trade_index
        return self.calendar[calendar_index - 1], self.calendar[calendar_index]

    def finished(self):
        return self.trade_index >= self.trade_len - 1

    def step(self):
        self.trade_index = self.trade_index + 1


class BaseEnv(BaseTradeCalendar):
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
        generate_report=False,
        verbose=False,
        **kwargs,
    ):
        self.generate_report = generate_report
        self.verbose = verbose
        super(BaseEnv, self).__init__(
            step_bar=step_bar, start_time=start_time, end_time=end_time, trade_account=trade_account, **kwargs
        )

    def reset(self, trade_account=None, **kwargs):
        super(BaseEnv, self).reset(**kwargs)
        if trade_account:
            self.trade_account = trade_account
            self.trade_account.reset(freq=self.step_bar, report=Report(), positions={})

    def get_init_state(self):
        init_state = {"current": self.trade_account.current}
        return init_state

    def execute(self, **kwargs):
        raise NotImplementedError("execute is not implemented!")

    def get_trade_account(self):
        raise NotImplementedError("get_trade_account is not implemented!")

    def get_report(self):
        raise NotImplementedError("get_report is not implemented!")


class SplitEnv(BaseEnv):
    def __init__(
        self,
        step_bar,
        sub_env,
        sub_strategy,
        start_time=None,
        end_time=None,
        trade_account=None,
        trade_exchange=None,
        generate_report=False,
        verbose=False,
        **kwargs,
    ):
        self.sub_env = sub_env
        self.sub_strategy = sub_strategy
        super(SplitEnv, self).__init__(
            step_bar=step_bar,
            start_time=start_time,
            end_time=end_time,
            trade_account=trade_account,
            trade_exchange=trade_exchange,
            generate_report=generate_report,
            verbose=verbose,
            **kwargs,
        )

    def reset(self, trade_account=None, trade_exchange=None, **kwargs):
        super(SplitEnv, self).reset(trade_account=trade_account, **kwargs)
        if trade_account:
            self.sub_env.reset(trade_account=copy.copy(trade_account))
        if trade_exchange:
            self.trade_exchange = trade_exchange

    def execute(self, order_list, **kwargs):
        if self.finished():
            raise StopIteration(f"this env has completed its task, please reset it if you want to call it!")
        # if self.track:
        #    yield action
        # episode_reward = 0
        super(SplitEnv, self).step()
        trade_start_time, trade_end_time = self._get_calendar_time(self.trade_index)
        self.sub_env.reset(start_time=trade_start_time, end_time=trade_end_time)
        self.sub_strategy.reset(start_time=trade_start_time, end_time=trade_end_time, trade_order_list=order_list)
        trade_state = self.sub_env.get_init_state()
        while not self.sub_env.finished():
            _order_list = self.sub_strategy.generate_order_list(**trade_state)
            trade_state, trade_info = self.sub_env.execute(order_list=_order_list)

        self.trade_account.update_bar_end(
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            trade_exchange=self.trade_exchange,
            update_report=self.generate_report,
        )
        _obs = {"current": self.trade_account.current}
        _info = {}
        return _obs, _info

    def get_report(self):
        sub_env_report_dict = self.sub_env.get_report()
        if self.generate_report:
            _report = self.trade_account.report.generate_report_dataframe()
            _positions = self.trade_account.get_positions()
            _count, _freq = parse_freq(self.step_bar)
            sub_env_report_dict.update({f"{_count}{_freq}": (_report, _positions)})
            return sub_env_report_dict
        else:
            return sub_env_report_dict


class SimulatorEnv(BaseEnv):
    def __init__(
        self,
        step_bar,
        start_time=None,
        end_time=None,
        trade_account=None,
        trade_exchange=None,
        generate_report=False,
        verbose=False,
        **kwargs,
    ):
        super(SimulatorEnv, self).__init__(
            step_bar=step_bar,
            start_time=start_time,
            end_time=end_time,
            trade_account=trade_account,
            trade_exchange=trade_exchange,
            generate_report=generate_report,
            verbose=verbose,
            **kwargs,
        )

    def reset(self, trade_exchange=None, **kwargs):
        super(SimulatorEnv, self).reset(**kwargs)
        if trade_exchange:
            self.trade_exchange = trade_exchange

    def execute(self, order_list, **kwargs):
        """
        Return: obs, done, info
        """
        if self.finished():
            raise StopIteration(f"this env has completed its task, please reset it if you want to call it!")
        super(SimulatorEnv, self).step()
        trade_start_time, trade_end_time = self._get_calendar_time(self.trade_index)
        trade_info = []
        for order in order_list:
            if self.trade_exchange.check_order(order) is True:
                # execute the order
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                    order, trade_account=self.trade_account
                )
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
        self.trade_account.update_bar_end(
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            trade_exchange=self.trade_exchange,
            update_report=self.generate_report,
        )
        _obs = {"current": self.trade_account.current}
        _info = {"trade_info": trade_info}
        return _obs, _info

    def get_report(self):
        if self.generate_report:
            _report = self.trade_account.report.generate_report_dataframe()
            _positions = self.trade_account.get_positions()
            _count, _freq = parse_freq(self.step_bar)
            return {f"{_count}{_freq}": (_report, _positions)}
        else:
            return {}
