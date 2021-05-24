# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import warnings
import pandas as pd

from .position import Position
from .report import Report
from .order import Order


"""
rtn & earning in the Account
    rtn:
        from order's view
        1.change if any order is executed, sell order or buy order
        2.change at the end of today,   (today_clse - stock_price) * amount
    earning
        from value of current position
        earning will be updated at the end of trade date
        earning = today_value - pre_value
    **is consider cost**
        while earning is the difference of two position value, so it considers cost, it is the true return rate
        in the specific accomplishment for rtn, it does not consider cost, in other words, rtn - cost = earning
"""


class Account:
    def __init__(self, init_cash, freq: str = "day", benchmark_config: dict = {}):
        self.init_vars(init_cash, freq, benchmark_config)

    def init_vars(self, init_cash, freq: str, benchmark_config: dict):

        # init cash
        self.init_cash = init_cash
        self.current = Position(cash=init_cash)
        self.reset(freq=freq, benchmark_config=benchmark_config, init_report=True)

    def reset_report(self, freq, benchmark_config):
        self.report = Report(freq, benchmark_config)
        self.positions = {}
        self.rtn = 0
        self.ct = 0
        self.to = 0
        self.val = 0
        self.earning = 0

    def reset(self, freq=None, benchmark_config=None, init_report=False):
        """reset freq and report of account

        Parameters
        ----------
        freq : str, optional
            frequency of account & report, by default None
        benchmark_config : {}, optional
            benchmark config of report, by default None
        init_report : bool, optional
            whether to initialize the report, by default False
        """
        if freq is not None:
            self.freq = freq
        if benchmark_config is not None:
            self.benchmark_config = benchmark_config

        if freq is not None or benchmark_config is not None or init_report:
            self.reset_report(self.freq, self.benchmark_config)

    def get_positions(self):
        return self.positions

    def get_cash(self):
        return self.current.position["cash"]

    def _update_state_from_order(self, order, trade_val, cost, trade_price):
        # update turnover
        self.to += trade_val
        # update cost
        self.ct += cost
        # update return
        # update self.rtn from order
        trade_amount = trade_val / trade_price
        if order.direction == Order.SELL:  # 0 for sell
            # when sell stock, get profit from price change
            profit = trade_val - self.current.get_stock_price(order.stock_id) * trade_amount
            self.rtn += profit  # note here do not consider cost
        elif order.direction == Order.BUY:  # 1 for buy
            # when buy stock, we get return for the rtn computing method
            # profit in buy order is to make self.rtn is consistent with self.earning at the end of date
            profit = self.current.get_stock_price(order.stock_id) * trade_amount - trade_val
            self.rtn += profit

    def update_order(self, order, trade_val, cost, trade_price):
        # if stock is sold out, no stock price information in Position, then we should update account first, then update current position
        # if stock is bought, there is no stock in current position, update current, then update account
        # The cost will be substracted from the cash at last. So the trading logic can ignore the cost calculation
        if order.direction == Order.SELL:
            # sell stock
            self._update_state_from_order(order, trade_val, cost, trade_price)
            # update current position
            # for may sell all of stock_id
            self.current.update_order(order, trade_val, cost, trade_price)
        else:
            # buy stock
            # deal order, then update state
            self.current.update_order(order, trade_val, cost, trade_price)
            self._update_state_from_order(order, trade_val, cost, trade_price)

    def update_bar_count(self):
        self.current.add_count_all(bar=self.freq)

    def update_bar_report(self, trade_start_time, trade_end_time, trade_exchange):
        """
        trade_start_time: pd.TimeStamp
        trade_end_time: pd.TimeStamp
        quote: pd.DataFrame (code, date), collumns
        when the end of trade date
        - update rtn
        - update price for each asset
        - update value for this account
        - update earning (2nd view of return )
        - update holding day, count of stock
        - update position hitory
        - update report
        :return: None
        """
        # update price for stock in the position and the profit from changed_price
        stock_list = self.current.get_stock_list()
        for code in stock_list:
            # if suspend, no new price to be updated, profit is 0
            if trade_exchange.check_stock_suspended(code, trade_start_time, trade_end_time):
                continue
            bar_close = trade_exchange.get_close(code, trade_start_time, trade_end_time)
            self.current.update_stock_price(stock_id=code, price=bar_close)
        # update holding day count

        # update value
        self.val = self.current.calculate_value()
        # update earning
        # account_value - last_account_value
        # for the first trade date, account_value - init_cash
        # self.report.is_empty() to judge is_first_trade_date
        # get last_account_value, now_account_value, now_stock_value
        if self.report.is_empty():
            last_account_value = self.init_cash
        else:
            last_account_value = self.report.get_latest_account_value()
        now_account_value = self.current.calculate_value()
        now_stock_value = self.current.calculate_stock_value()
        self.earning = now_account_value - last_account_value
        # update report for today
        # judge whether the the trading is begin.
        # and don't add init account state into report, due to we don't have excess return in those days.
        self.report.update_report_record(
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            account_value=now_account_value,
            cash=self.current.position["cash"],
            return_rate=(self.earning + self.ct) / last_account_value,
            # here use earning to calculate return, position's view, earning consider cost, true return
            # in order to make same definition with original backtest in evaluate.py
            turnover_rate=self.to / last_account_value,
            cost_rate=self.ct / last_account_value,
            stock_value=now_stock_value,
        )
        # set now_account_value to position
        self.current.position["now_account_value"] = now_account_value
        self.current.update_weight_all()
        # update positions
        # note use deepcopy
        self.positions[trade_start_time] = copy.deepcopy(self.current)

        # finish today's updation
        # reset the bar variables
        self.rtn = 0
        self.ct = 0
        self.to = 0
