# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import numpy as np
import pandas as pd

from ..data.dataset import DatasetH
from ..backtest.order import Order
from .order_generator import OrderGenWInteract

"""
1. BaseStrategy 的粒度一定是数据粒度的整数倍
- 关于calendar的合并咋整
- adjust_dates这个东西啥用
- label和freq和strategy的bar分离，这个如何决策呢
"""
class BaseStrategy:
    def __init__(self, bar, start_time, end_time):
        self.bar = bar
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time
    
    def generate_action(self, current):
        pass


class RuleStrategy(BaseStrategy):
    pass

class DLStrategy(BaseStrategy):
    def __init__(self, bar, model, dataset:DatasetH, start_time=None, end_time=None):
        super(DLStrategy, self).__init__(bar, start_time, end_time)
        self.model = model
        self.dataset = dataset
        self.pred_score_all = self.model.predict(dataset)
        self.pred_score = None
        _pred_dates = pred.index.get_level_values(level="datetime")
        self.start_time = _pred_dates.min() if start_time is None else start_time
        self.end_time = _pred_dates.max() if end_time is None else end_time
        self.pred_date = [pd.Timestamp(self.start_time), *D.calendar(start_time=_pred_dates.min(), end_time=_pred_dates.max(), freq=bar), self.end_time]
        self.current_index = -1
        self.pred_length = len(self.pred_date)

     def _update_pred_score(self):
        """update pred score
        """
        pass

class AdjustTimer:
    """AdjustTimer
    Responsible for timing of position adjusting

    This is designed as multiple inheritance mechanism due to:
    - the is_adjust may need access to the internel state of a strategy.

    - it can be reguard as a enhancement to the existing strategy.
    """

    # adjust position in each trade date
    def is_adjust(self, trade_date):
        """is_adjust
        Return if the strategy can adjust positions on `trade_date`
        Will normally be used in strategy do trading with trade frequency
        """
        return True


class ListAdjustTimer(AdjustTimer):
    def __init__(self, adjust_dates=None):
        """__init__

        :param adjust_dates: an iterable object, it will return a timelist for trading dates
        """
        if adjust_dates is None:
            # None indicates that all dates is OK for adjusting
            self.adjust_dates = None
        else:
            self.adjust_dates = {pd.Timestamp(dt) for dt in adjust_dates}

    def is_adjust(self, trade_date):
        if self.adjust_dates is None:
            return True
        return pd.Timestamp(trade_date) in self.adjust_dates

class TopkDropoutStrategy(DLStrategy, ListAdjustTimer):
    def __init__(
        self,
        bar,
        model,
        dataset,
        trade_exchange,
        topk,
        n_drop,
        start_time=None,
        end_time=None,
        method_sell="bottom",
        method_buy="top",
        risk_degree=0.95,
        thresh=1,
        hold_thresh=1,
        only_tradable=False,
        **kwargs,
    ):
        """
        Parameters
        -----------
        topk : int
            the number of stocks in the portfolio.
        n_drop : int
            number of stocks to be replaced in each trading date.
        method_sell : str
            dropout method_sell, random/bottom.
        method_buy : str
            dropout method_buy, random/top.
        risk_degree : float
            position percentage of total value.
        thresh : int
            minimun holding days since last buy singal of the stock.
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.
            if only_tradable:
                strategy will make buy sell decision without checking the tradable state of the stock.
            else:
                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.
        """
        super(TopkDropoutStrategy, self).__init__(bar, model, dataset, start_time, end_time)
        ListAdjustTimer.__init__(self, kwargs.get("adjust_dates", None))
        self.trade_exchange = trade_exchange
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.risk_degree = risk_degree
        self.thresh = thresh
        # self.stock_count['code'] will be the days the stock has been hold
        # since last buy signal. This is designed for thresh
        self.stock_count = {}

        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
    
    def get_risk_degree(self, date):
        """get_risk_degree
        Return the proportion of your total value you will used in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amoutn of your total value by default
        return self.risk_degree

    def generate_action(self, current):

        self.current_index += 1
        
        if not self.is_adjust(trade_date):
            return []

        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            def get_first_n(l, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(l) if reverse else l:
                    if self.trade_exchange.is_stock_tradable(stock_id=si, trade_date=trade_date):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(l, n):
                return get_first_n(l, n, reverse=True)

            def filter_stock(l):
                return [si for si in l if self.trade_exchange.is_stock_tradable(stock_id=si, trade_date=trade_date)]

        else:
            # Otherwise, the stock will make decision with out the stock tradable info
            def get_first_n(l, n):
                return list(l)[:n]

            def get_last_n(l, n):
                return list(l)[-n:]

            def filter_stock(l):
                return l

        current_temp = copy.deepcopy(current)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        # last position (sorted by score)
        last = self.pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            today = get_first_n(
                self.pred_score[~self.pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(self.pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")
        # combine(new stocks + last stocks),  we will drop stocks from this list
        # In case of dropping higher score stock and buying lower score stock.
        comb = self.pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        # Get the stock list we really want to sell (After filtering the case that we sell high and buy low)
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:  #  No enough candidates
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # Get the stock list we really want to buy
        buy = today[: len(sell) + self.topk - len(last)]

        # buy singal: if a stock falls into topk, it appear in the buy_sinal
        buy_signal = self.pred_score.sort_values(ascending=False).iloc[: self.topk].index

        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(stock_id=code, trade_date=trade_date):
                continue
            if code in sell:
                # check hold limit
                if self.stock_count[code] < self.thresh or current_temp.get_stock_count(code) < self.hold_thresh:
                    # can not sell this code
                    # no buy signal, but the stock is kept
                    self.stock_count[code] += 1
                    continue
                # sell order
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    trade_date=trade_date,
                    direction=Order.SELL,  # 0 for sell, 1 for buy
                    factor=self.trade_exchange.get_factor(code, trade_date),
                )
                # is order executable
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(sell_order, position=current_temp)
                    # update cash
                    cash += trade_val - trade_cost
                    # sold
                    del self.stock_count[code]
                else:
                    # no buy signal, but the stock is kept
                    self.stock_count[code] += 1
            elif code in buy_signal:
                # NOTE: This is different from the original version
                # get new buy signal
                # Only the stock fall in to topk will produce buy signal
                self.stock_count[code] = 1
            else:
                self.stock_count[code] += 1
        # buy new stock
        # note the current has been changed
        current_stock_list = current_temp.get_stock_list()
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
        # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock suspended
            if not self.trade_exchange.is_stock_tradable(stock_id=code, trade_date=trade_date):
                continue
            # buy order
            buy_price = self.trade_exchange.get_deal_price(stock_id=code, trade_date=trade_date)
            buy_amount = value / buy_price
            factor = self.trade_exchange.quote[(code, trade_date)]["$factor"]
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                trade_date=trade_date,
                direction=Order.BUY,  # 1 for buy
                factor=factor,
            )
            buy_order_list.append(buy_order)
            self.stock_count[code] = 1
        return sell_order_list + buy_order_list
