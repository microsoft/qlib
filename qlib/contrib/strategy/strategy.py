# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import numpy as np
import pandas as pd

from ..backtest.order import Order
from ...utils import get_pre_trading_date
from .order_generator import OrderGenWInteract


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
        """Parameter
        score_series : pd.Seires
            stock_id , score
        current : Position()
            current state of position
            DO NOT directly change the state of current
        trade_exchange : Exchange()
            trade exchange
        pred_date : pd.Timestamp
            predict date
        trade_date : pd.Timestamp
            trade date

        DO NOT directly change the state of current
        """
        pass

    def update(self, score_series, pred_date, trade_date):
        """User can use this method to update strategy state each trade date.
        Parameter
        ---------
        score_series : pd.Series
            stock_id , score
        pred_date : pd.Timestamp
            oredict date
        trade_date : pd.Timestamp
            trade date
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
                mode : model used in 'online' module
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

        :param inner_strategy: set the inner strategy
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

    This is designed as multiple inheritance mechanism due to
    - the is_adjust may need access to the internel state of a strategyw
    - it can be reguard as a enhancement to the existing strategy
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
        """Parameter:
        score : pred score for this trade date, pd.Series, index is stock_id, contain 'score' column
        current : current position, use Position() class
        trade_exchange : Exchange()
        trade_date : trade date
        generate target position from score for this date and the current position
        The cash is not considered in the position
        """
        raise NotImplementedError()

    def generate_order_list(self, score_series, current, trade_exchange, pred_date, trade_date):
        """Parameter
        score_series : pd.Seires
            stock_id , score
        current : Position()
            current of account
        trade_exchange : Exchange()
            exchange
        trade_date : pd.Timestamp
            date
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
    def __init__(self, topk, n_drop, method="bottom", risk_degree=0.95, thresh=1, hold_thresh=1, **kwargs):
        """Parameter
        topk : int
            The number of stocks in the portfolio
        n_drop : int
            number of stocks to be replaced in each trading date
        method : str
            dropout method, random/bottom
        risk_degree : float
            position percentage of total value
        thresh : int
            minimun holding days since last buy singal of the stock
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.thresh
        """
        super(TopkDropoutStrategy, self).__init__()
        ListAdjustTimer.__init__(self, kwargs.get("adjust_dates", None))
        self.topk = topk
        self.n_drop = n_drop
        self.method = method
        self.risk_degree = risk_degree
        self.thresh = thresh
        # self.stock_count['code'] will be the days the stock has been hold
        # since last buy signal. This is designed for thresh
        self.stock_count = {}

        self.hold_thresh = hold_thresh

    def get_risk_degree(self, date):
        """get_risk_degree
        Return the proportion of your total value you will used in investment.
        Dynamically risk_degree will result in Market timing
        """
        # It will use 95% amoutn of your total value by default
        return self.risk_degree

    def generate_order_list(self, score_series, current, trade_exchange, pred_date, trade_date):
        """Gnererate order list according to score_series at trade_date.
            will not change current.
        Parameter
            score_series : pd.Seires
                stock_id , score
            current : Position()
                current of account
            trade_exchange : Exchange()
                exchange
            pred_date : pd.Timestamp
                predict date
            trade_date : pd.Timestamp
                trade date
        """
        if not self.is_adjust(trade_date):
            return []
        current_temp = copy.deepcopy(current)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        last = score_series.reindex(current_stock_list).sort_values(ascending=False).index
        today = (
            score_series[~score_series.index.isin(last)]
            .sort_values(ascending=False)
            .index[: self.n_drop + self.topk - len(last)]
        )
        comb = score_series.reindex(last.union(today)).sort_values(ascending=False).index
        if self.method == "bottom":
            sell = last[last.isin(comb[-self.n_drop :])]
        elif self.method == "random":
            sell = pd.Index(np.random.choice(last, self.n_drop) if len(last) else [])
        buy = today[: len(sell) + self.topk - len(last)]
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
            elif code in buy:
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
