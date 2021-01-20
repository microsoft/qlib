# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import numpy as np
import pandas as pd

from ..backtest.order import Order
from ...utils import get_pre_trading_date
from .order_generator import OrderGenWInteract


# TODO: The base strategies will be moved out of contrib to core code
class BaseStrategy:
    def __init__(self):
        pass

    def get_risk_degree(self, date):
        """get_risk_degree
        Return the proportion of your total value you will used in investment.
        Dynamically risk_degree will result in Market timing
        """
        # It will use 95% amount of your total value by default
        return 0.95

    def generate_order_list(self, score_series, current, trade_exchange, pred_date, trade_date):
        """
        DO NOT directly change the state of current

        Parameters
        -----------
        score_series : pd.Series
            stock_id , score.
        current : Position()
            current state of position.
            DO NOT directly change the state of current.
        trade_exchange : Exchange()
            trade exchange.
        pred_date : pd.Timestamp
            predict date.
        trade_date : pd.Timestamp
            trade date.
        """
        pass

    def update(self, score_series, pred_date, trade_date):
        """User can use this method to update strategy state each trade date.
        Parameters
        -----------
        score_series : pd.Series
            stock_id , score.
        pred_date : pd.Timestamp
            oredict date.
        trade_date : pd.Timestamp
            trade date.
        """
        pass

    def init(self, **kwargs):
        """Some strategy need to be initial after been implemented,
        User can use this method to init his strategy with parameters needed.
        """
        pass

    def get_init_args_from_model(self, model, init_date):
        """
        This method only be used in 'online' module, it will generate the *args to initial the strategy.
            :param
                mode : model used in 'online' module.
        """
        return {}


class StrategyWrapper:
    """
    StrategyWrapper is a wrapper of another strategy.
    By overriding some methods to make some changes on the basic strategy
    Cost control and risk control will base on this class.
    """

    def __init__(self, inner_strategy):
        """__init__

        :param inner_strategy: set the inner strategy.
        """
        self.inner_strategy = inner_strategy

    def __getattr__(self, name):
        """__getattr__

        :param name: If no implementation in this method. Call the method in the innter_strategy by default.
        """
        return getattr(self.inner_strategy, name)


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


class WeightStrategyBase(BaseStrategy, AdjustTimer):
    def __init__(self, order_generator_cls_or_obj=OrderGenWInteract, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator = order_generator_cls_or_obj()
        else:
            self.order_generator = order_generator_cls_or_obj

    def generate_target_weight_position(self, score, current, trade_date):
        """
        Generate target position from score for this date and the current position.The cash is not considered in the position

        Parameters
        -----------
        score : pd.Series
            pred score for this trade date, index is stock_id, contain 'score' column.
        current : Position()
            current position.
        trade_exchange : Exchange()
        trade_date : pd.Timestamp
            trade date.
        """
        raise NotImplementedError()

    def generate_order_list(self, score_series, current, trade_exchange, pred_date, trade_date):
        """
        Parameters
        -----------
        score_series : pd.Seires
            stock_id , score.
        current : Position()
            current of account.
        trade_exchange : Exchange()
            exchange.
        trade_date : pd.Timestamp
            date.
        """
        # judge if to adjust
        if not self.is_adjust(trade_date):
            return []
        # generate_order_list
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list
        current_temp = copy.deepcopy(current)
        target_weight_position = self.generate_target_weight_position(
            score=score_series, current=current_temp, trade_date=trade_date
        )

        order_list = self.order_generator.generate_order_list_from_target_weight_position(
            current=current_temp,
            trade_exchange=trade_exchange,
            risk_degree=self.get_risk_degree(trade_date),
            target_weight_position=target_weight_position,
            pred_date=pred_date,
            trade_date=trade_date,
        )
        return order_list


class TopkDropoutStrategy(BaseStrategy, ListAdjustTimer):
    def __init__(
        self,
        topk,
        n_drop,
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
        super(TopkDropoutStrategy, self).__init__()
        ListAdjustTimer.__init__(self, kwargs.get("adjust_dates", None))
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

    def generate_order_list(self, score_series, current, trade_exchange, pred_date, trade_date):
        """
        Gnererate order list according to score_series at trade_date, will not change current.

        Parameters
        -----------
        score_series : pd.Series
            stock_id , score.
        current : Position()
            current of account.
        trade_exchange : Exchange()
            exchange.
        pred_date : pd.Timestamp
            predict date.
        trade_date : pd.Timestamp
            trade date.
        """
        if not self.is_adjust(trade_date):
            return []

        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            def get_first_n(l, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(l) if reverse else l:
                    if trade_exchange.is_stock_tradable(stock_id=si, trade_date=trade_date):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(l, n):
                return get_first_n(l, n, reverse=True)

            def filter_stock(l):
                return [si for si in l if trade_exchange.is_stock_tradable(stock_id=si, trade_date=trade_date)]

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
        last = score_series.reindex(current_stock_list).sort_values(ascending=False).index
        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            today = get_first_n(
                score_series[~score_series.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(score_series.sort_values(ascending=False).index, self.topk)
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
        comb = score_series.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

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
        buy_signal = score_series.sort_values(ascending=False).iloc[: self.topk].index

        for code in current_stock_list:
            if not trade_exchange.is_stock_tradable(stock_id=code, trade_date=trade_date):
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
                    factor=trade_exchange.get_factor(code, trade_date),
                )
                # is order executable
                if trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = trade_exchange.deal_order(sell_order, position=current_temp)
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

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not consider it
        # as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock supended
            if not trade_exchange.is_stock_tradable(stock_id=code, trade_date=trade_date):
                continue
            # buy order
            buy_price = trade_exchange.get_deal_price(stock_id=code, trade_date=trade_date)
            buy_amount = value / buy_price
            factor = trade_exchange.quote[(code, trade_date)]["$factor"]
            buy_amount = trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
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
