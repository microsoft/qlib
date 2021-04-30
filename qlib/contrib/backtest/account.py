# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import pandas as pd

from .position import Position
from .report import Report
from .order import Order
from ...data import D
from ...utils import parse_freq, sample_feature


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
    def __init__(self, init_cash, benchmark=None, start_time=None, end_time=None, freq=None):
        self.init_vars(init_cash, benchmark, start_time, end_time)

    def init_vars(self, init_cash, benchmark=None, start_time=None, end_time=None, freq=None):
        """
        Parameters
        ----------
        - benchmark: str/list/pd.Series
            `benchmark` is pd.Series, `index` is trading date; the value T is the change from T-1 to T.
                example:
                    print(D.features(D.instruments('csi500'), ['$close/Ref($close, 1)-1'])['$close/Ref($close, 1)-1'].head())
                        2017-01-04    0.011693
                        2017-01-05    0.000721
                        2017-01-06   -0.004322
                        2017-01-09    0.006874
                        2017-01-10   -0.003350
            `benchmark` is list, will use the daily average change of the stock pool in the list as the 'bench'.
            `benchmark` is str, will use the daily change as the 'bench'.
        benchmark code, default is SH000905 CSI500

        """
        # init cash
        self.init_cash = init_cash
        self.benchmark = benchmark
        self.start_time = start_time
        self.end_time = end_time
        self.freq = freq
        self.current = Position(cash=init_cash)
        self.positions = {}
        self.rtn = 0
        self.ct = 0
        self.to = 0
        self.val = 0
        self.earning = 0
        self.report = Report()
        if freq and benchmark:
            self.bench = self._cal_benchmark(benchmark, start_time, end_time, freq)

    def _cal_benchmark(self, benchmark, start_time=None, end_time=None, freq=None):
        if isinstance(benchmark, pd.Series):
            return benchmark
        else:
            if freq is None:
                raise ValueError("benchmark freq can't be None!")
            _codes = benchmark if isinstance(benchmark, list) else [benchmark]
            fields = ["$close/Ref($close,1)-1"]
            try:
                _temp_result = D.features(_codes, fields, start_time, end_time, freq=freq, disk_cache=1)
            except ValueError:
                _, norm_freq = parse_freq(freq)
                if norm_freq in ["month", "week", "day"]:
                    try:
                        _temp_result = D.features(_codes, fields, start_time, end_time, freq="day", disk_cache=1)
                    except ValueError:
                        _temp_result = D.features(_codes, fields, start_time, end_time, freq="minute", disk_cache=1)
                elif norm_freq == "minute":
                    _temp_result = D.features(_codes, fields, start_time, end_time, freq="minute", disk_cache=1)
                else:
                    raise ValueError(f"benchmark freq {freq} is not supported")
            if len(_temp_result) == 0:
                raise ValueError(f"The benchmark {_codes} does not exist. Please provide the right benchmark")
            return _temp_result.groupby(level="datetime")[_temp_result.columns.tolist()[0]].mean().fillna(0)

    def _sample_benchmark(self, bench, trade_start_time, trade_end_time):
        def cal_change(x):
            return x.prod() - 1

        _ret = sample_feature(bench, trade_start_time, trade_end_time, method=cal_change)
        return 0 if _ret is None else _ret

    def reset(self, benchmark=None, freq=None, **kwargs):
        if benchmark:
            self.benchmark = benchmark
        if freq:
            self.freq = freq
        if self.freq and self.benchmark and (freq or benchmark):
            self.bench = self._cal_benchmark(self.benchmark, self.start_time, self.end_time, self.freq)

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_positions(self):
        return self.positions

    def get_cash(self):
        return self.current.position["cash"]

    def update_state_from_order(self, order, trade_val, cost, trade_price):
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
        trade_amount = trade_val / trade_price
        if order.direction == Order.SELL:
            # sell stock
            self.update_state_from_order(order, trade_val, cost, trade_price)
            # update current position
            # for may sell all of stock_id
            self.current.update_order(order, trade_val, cost, trade_price)
        else:
            # buy stock
            # deal order, then update state
            self.current.update_order(order, trade_val, cost, trade_price)
            self.update_state_from_order(order, trade_val, cost, trade_price)

    def update_bar_end(self, trade_start_time, trade_end_time, trade_exchange, update_report):
        """
        start_time: pd.TimeStamp
        end_time: pd.TimeStamp
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
        self.current.add_count_all(bar=self.freq)
        if update_report is None:
            return
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
            trade_time=trade_start_time,
            account_value=now_account_value,
            cash=self.current.position["cash"],
            return_rate=(self.earning + self.ct) / last_account_value,
            # here use earning to calculate return, position's view, earning consider cost, true return
            # in order to make same definition with original backtest in evaluate.py
            turnover_rate=self.to / last_account_value,
            cost_rate=self.ct / last_account_value,
            stock_value=now_stock_value,
            bench_value=self._sample_benchmark(self.bench, trade_start_time, trade_end_time),
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

    def load_account(self, account_path):
        report = Report()
        position = Position()
        report.load_report(account_path / "report.csv")
        position.load_position(account_path / "position.xlsx")

        # assign values
        self.init_vars(position.init_cash)
        self.current = position
        self.report = report

    def save_account(self, account_path):
        self.current.save_position(account_path / "position.xlsx")
        self.report.save_report(account_path / "report.csv")
