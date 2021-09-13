# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import copy
from typing import Dict, List, Tuple, TYPE_CHECKING
from qlib.utils import init_instance_by_config
import warnings
import pandas as pd

from .position import BasePosition, InfPosition, Position
from .report import Report, Indicator
from .order import BaseTradeDecision, Order

if TYPE_CHECKING:
    from .exchange import Exchange

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


class AccumulatedInfo:
    """accumulated trading info, including accumulated return\cost\turnover"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rtn = 0  # accumulated return, do not consider cost
        self.cost = 0  # accumulated cost
        self.to = 0  # accumulated turnover

    def add_return_value(self, value):
        self.rtn += value

    def add_cost(self, value):
        self.cost += value

    def add_turnover(self, value):
        self.to += value

    @property
    def get_return(self):
        return self.rtn

    @property
    def get_cost(self):
        return self.cost

    @property
    def get_turnover(self):
        return self.to


class Account:
    def __init__(
        self,
        init_cash: float = 1e9,
        position_dict: dict = {},
        freq: str = "day",
        benchmark_config: dict = {},
        pos_type: str = "Position",
        port_metr_enabled: bool = True,
    ):
        """the trade account of backtest.

        Parameters
        ----------
        init_cash : float, optional
            initial cash, by default 1e9
        position_dict : Dict[
                            stock_id,
                            Union[
                                int,  # it is equal to {"amount": int}
                                {"amount": int, "price"(optional): float},
                            ]
                        ]
            initial stocks with parameters amount and price,
            if there is no price key in the dict of stocks, it will be filled by _fill_stock_value.
            by default {}.
        """

        self._pos_type = pos_type
        self._port_metr_enabled = port_metr_enabled

        self.init_cash = init_cash
        self.current: BasePosition = init_instance_by_config(
            {
                "class": self._pos_type,
                "kwargs": {
                    "cash": init_cash,
                    "position_dict": position_dict,
                },
                "module_path": "qlib.backtest.position",
            }
        )
        self.report = None
        self.positions = {}

        # in of reset ignore None values
        self.benchmark_config = benchmark_config
        self.freq = freq

        self.reset(freq=freq, benchmark_config=benchmark_config, init_report=True)

    def is_port_metr_enabled(self):
        """
        Is portfolio-based metrics enabled.
        """
        return self._port_metr_enabled and not self.current.skip_update()

    def reset_report(self, freq, benchmark_config):
        # portfolio related metrics
        if self.is_port_metr_enabled():
            self.accum_info = AccumulatedInfo()
            self.report = Report(freq, benchmark_config)
            self.positions = {}
            # fill stock value
            # The frequency of account may not align with the trading frequency.
            # This may result in obscure bugs when data quality is low.
            if isinstance(self.benchmark_config, dict) and self.benchmark_config.get("start_time") is not None:
                self.current.fill_stock_value(self.benchmark_config["start_time"], self.freq)

        # trading related metrics(e.g. high-frequency trading)
        self.indicator = Indicator()

    def reset(self, freq=None, benchmark_config=None, init_report=False, port_metr_enabled: bool = None):
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

        if port_metr_enabled is not None:
            self._port_metr_enabled = port_metr_enabled

        if freq is not None or benchmark_config is not None or init_report:
            self.reset_report(self.freq, self.benchmark_config)

    def get_positions(self):
        return self.positions

    def get_cash(self):
        return self.current.get_cash()

    def _update_accum_info_from_order(self, order, trade_val, cost, trade_price):
        if self.is_port_metr_enabled():
            # update turnover
            self.accum_info.add_turnover(trade_val)
            # update cost
            self.accum_info.add_cost(cost)

            # update return from order
            trade_amount = trade_val / trade_price
            if order.direction == Order.SELL:  # 0 for sell
                # when sell stock, get profit from price change
                profit = trade_val - self.current.get_stock_price(order.stock_id) * trade_amount
                self.accum_info.add_return_value(profit)  # note here do not consider cost

            elif order.direction == Order.BUY:  # 1 for buy
                # when buy stock, we get return for the rtn computing method
                # profit in buy order is to make rtn is consistent with earning at the end of bar
                profit = self.current.get_stock_price(order.stock_id) * trade_amount - trade_val
                self.accum_info.add_return_value(profit)  # note here do not consider cost

    def update_order(self, order, trade_val, cost, trade_price):
        if self.current.skip_update():
            # TODO: supporting polymorphism for account
            # updating order for infinite position is meaningless
            return

        # if stock is sold out, no stock price information in Position, then we should update account first, then update current position
        # if stock is bought, there is no stock in current position, update current, then update account
        # The cost will be substracted from the cash at last. So the trading logic can ignore the cost calculation
        if order.direction == Order.SELL:
            # sell stock
            self._update_accum_info_from_order(order, trade_val, cost, trade_price)
            # update current position
            # for may sell all of stock_id
            self.current.update_order(order, trade_val, cost, trade_price)
        else:
            # buy stock
            # deal order, then update state
            self.current.update_order(order, trade_val, cost, trade_price)
            self._update_accum_info_from_order(order, trade_val, cost, trade_price)

    def update_bar_count(self):
        """at the end of the trading bar, update holding bar, count of stock"""
        # update holding day count
        # NOTE: updating bar_count does not only serve portfolio metrics, it also serve the strategy
        if not self.current.skip_update():
            self.current.add_count_all(bar=self.freq)

    def update_current(self, trade_start_time, trade_end_time, trade_exchange):
        """update current to make rtn consistent with earning at the end of bar"""
        # update price for stock in the position and the profit from changed_price
        # NOTE: updating position does not only serve portfolio metrics, it also serve the strategy
        if not self.current.skip_update():
            stock_list = self.current.get_stock_list()
            for code in stock_list:
                # if suspend, no new price to be updated, profit is 0
                if trade_exchange.check_stock_suspended(code, trade_start_time, trade_end_time):
                    continue
                bar_close = trade_exchange.get_close(code, trade_start_time, trade_end_time)
                self.current.update_stock_price(stock_id=code, price=bar_close)

    def update_report(self, trade_start_time, trade_end_time):
        """update position history, report"""
        # calculate earning
        # account_value - last_account_value
        # for the first trade date, account_value - init_cash
        # self.report.is_empty() to judge is_first_trade_date
        # get last_account_value, last_total_cost, last_total_turnover
        if self.report.is_empty():
            last_account_value = self.init_cash
            last_total_cost = 0
            last_total_turnover = 0
        else:
            last_account_value = self.report.get_latest_account_value()
            last_total_cost = self.report.get_latest_total_cost()
            last_total_turnover = self.report.get_latest_total_turnover()
        # get now_account_value, now_stock_value, now_earning, now_cost, now_turnover
        now_account_value = self.current.calculate_value()
        now_stock_value = self.current.calculate_stock_value()
        now_earning = now_account_value - last_account_value
        now_cost = self.accum_info.get_cost - last_total_cost
        now_turnover = self.accum_info.get_turnover - last_total_turnover
        # update report for today
        # judge whether the the trading is begin.
        # and don't add init account state into report, due to we don't have excess return in those days.
        self.report.update_report_record(
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            account_value=now_account_value,
            cash=self.current.position["cash"],
            return_rate=(now_earning + now_cost) / last_account_value,
            # here use earning to calculate return, position's view, earning consider cost, true return
            # in order to make same definition with original backtest in evaluate.py
            total_turnover=self.accum_info.get_turnover,
            turnover_rate=now_turnover / last_account_value,
            total_cost=self.accum_info.get_cost,
            cost_rate=now_cost / last_account_value,
            stock_value=now_stock_value,
        )
        # set now_account_value to position
        self.current.position["now_account_value"] = now_account_value
        self.current.update_weight_all()
        # update positions
        # note use deepcopy
        self.positions[trade_start_time] = copy.deepcopy(self.current)

    def update_bar_end(
        self,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        trade_exchange: Exchange,
        atomic: bool,
        outer_trade_decision: BaseTradeDecision,
        trade_info: list = None,
        inner_order_indicators: List[Dict[str, pd.Series]] = None,
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]] = None,
        indicator_config: dict = {},
    ):
        """update account at each trading bar step

        Parameters
        ----------
        trade_start_time : pd.Timestamp
            closed start time of step
        trade_end_time : pd.Timestamp
            closed end time of step
        trade_exchange : Exchange
            trading exchange, used to update current
        atomic : bool
            whether the trading executor is atomic, which means there is no higher-frequency trading executor inside it
            - if atomic is True, calculate the indicators with trade_info
            - else, aggregate indicators with inner indicators
        trade_info : List[(Order, float, float, float)], optional
            trading information, by default None
            - necessary if atomic is True
            - list of tuple(order, trade_val, trade_cost, trade_price)
        inner_order_indicators : Indicator, optional
            indicators of inner executor, by default None
            - necessary if atomic is False
            - used to aggregate outer indicators
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]] = None,
            The decision list of the inner level: List[Tuple[<decision>, <start_time>, <end_time>]]
            The inner level
        indicator_config : dict, optional
            config of calculating indicators, by default {}
        """
        if atomic is True and trade_info is None:
            raise ValueError("trade_info is necessary in atomic executor")
        elif atomic is False and inner_order_indicators is None:
            raise ValueError("inner_order_indicators is necessary in un-atomic executor")

        # TODO:  `update_bar_count` and  `update_current` should placed in Position and be merged.
        self.update_bar_count()
        self.update_current(trade_start_time, trade_end_time, trade_exchange)
        if self.is_port_metr_enabled():
            # report is portfolio related analysis
            self.update_report(trade_start_time, trade_end_time)

        # TODO: will skip empty decisions make it faster?  `outer_trade_decision.empty():`

        # indicator is trading (e.g. high-frequency order execution) related analysis
        self.indicator.reset()

        # aggregate the information for each order
        if atomic:
            self.indicator.update_order_indicators(trade_info)
        else:
            self.indicator.agg_order_indicators(
                inner_order_indicators,
                decision_list=decision_list,
                outer_trade_decision=outer_trade_decision,
                trade_exchange=trade_exchange,
                indicator_config=indicator_config,
            )

        # aggregate all the order metrics a single step
        self.indicator.cal_trade_indicators(trade_start_time, self.freq, indicator_config)

        # record the metrics
        self.indicator.record(trade_start_time)

    def get_report(self):
        """get the history report and postions instance"""
        if self.is_port_metr_enabled():
            _report = self.report.generate_report_dataframe()
            _positions = self.get_positions()
            return _report, _positions
        else:
            raise ValueError("generate_report should be True if you want to generate report")

    def get_trade_indicator(self) -> Indicator:
        """get the trade indicator instance, which has pa/pos/ffr info."""
        return self.indicator
