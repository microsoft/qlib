# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import numpy as np
import pandas as pd

from ..backtest.order import Order
from ...utils import get_pre_trading_date
from .order_generator import OrderGenWInteract


class BaseStrategy:
    """
    # Strategy framework document

    class Strategy(BaseStrategy):

        def __init__(self):
            # init for strategy
            super(Strategy, self).__init__()
            pass

        def generate_target_weight_position(self, score, current, trade_exchange, topk, buffer_margin, trade_date, risk_degree):
            '''Parameter:
            score : pred score for this trade date, pd.Series, index is stock_id, contain 'score' column
            current : current position, use Position() class
            trade_exchange : Exchange()
            topk : topk
            buffer_margin : buffer margin
            trade_date : trade date
            risk_degree : 0-1, 0.95 for example, use 95% money to trade
            :return
                target weight position
            generate target position from score for this date and the current position

            '''
            # strategy 1 ：select top k stocks by model scores, then equal-weight
            new_stock_list = list(score.sort_values(ascending=False).iloc[:topk].index)
            target_weight_position = {code: 1 / topk for code in new_stock_list}

            # strategy 2：select top buffer_margin stock as the buffer, for stock in current position: keep if in buffer, sell if not; then buy new stocks
            buffer = score.sort_values(ascending=False).iloc[:buffer_margin]
            current_stock_list = current.get_stock_list()
            mask = buffer.index.isin(current_stock_list)
            keep = set(buffer[mask].index)
            new_stock_list = list(keep) + list(buffer[~mask].index[:topk-len(keep)])
            target_weight_position = {code : 1/topk for code in new_stock_list}

            return target_weight_position

        def generate_target_amount_position(self, score, current, target_weight_position ,topk, buffer_margin, trade_exchange, trade_date, risk_degree):
            '''
            score : pred score for this trade date, pd.Series, index is stock_id, contain 'score' column
            current : current position, use Position() class
            target_weight_position : {stock_id : weight}
            trade_exchange : Exchange()
            topk : topk
            buffer_margin : buffer margin
            trade_date : trade date
            risk_degree : 0-1, 0.95 for example, use 95% money to trade
            :return:
            '''
            # strategy 1
            # parameters :
            #         topk : int, select topk stocks
            #         buffer_margin : size of buffer margin
            #
            #     description :
            #         hold topk stocks at each trade date
            #         when adjust position
            #             the score model will generate scores for each stock
            #             if the stock of current position not in top buffer_margin score, sell them out;
            #             then equally buy recommended stocks
            target_amount_dict = {}
            current_amount_dict = current.get_stock_amount_dict()
            buffer = score.sort_values(ascending=False).iloc[:buffer_margin]
            mask = buffer.index.isin(current_amount_dict)
            keep = set(buffer[mask].index)
            buy_stock_list = list(buffer[~mask].index[:topk - len(keep)])
            buy_cash = 0
            # calculate cash for buy order
            for stock_id in current_amount_dict:
                if stock_id in keep:
                    target_amount_dict[stock_id] = current_amount_dict[stock_id]
                else:
                    # stock_id not in keep
                    # trade check
                    # assume all of them can be sold out
                    if trade_exchange.check_stock_suspended(stock_id=stock_id, trade_date=trade_date):
                        pass
                    else:
                        buy_cash += current_amount_dict[stock_id] * trade_exchange.get_deal_price(stock_id=stock_id, trade_date=trade_date)
            # update close cost
            buy_cash /= (1 + trade_exchange.close_cost)
            # update cash
            buy_cash += current.get_cash()
            # update open cost
            buy_cash /= (1 + trade_exchange.open_cost)
            # consider risk degree
            buy_cash *= risk_degree
            # equally assigned
            value = buy_cash / len(buy_stock_list)
            # equally assigned
            value = buy_cash / len(buy_stock_list)
            for stock_id in buy_stock_list:
                if trade_exchange.check_stock_suspended(stock_id=stock_id, trade_date=trade_date):
                    pass
                else:
                    target_amount_dict[stock_id] = value / trade_exchange.get_deal_price(stock_id=stock_id, trade_date=trade_date) // trade_exchange.trade_unit * trade_exchange.trade_unit
            return target_amount_dict


            # strategy 2 : use trade_exchange.generate_amount_position_from_weight_position()
            # calculate value for current position
            current_amount_dict = current.get_stock_amount_dict()
            current_tradable_value = trade_exchange.calculate_amount_position_value(amount_dict=current_amount_dict,
                                                                                      trade_date=trade_date, only_tradable=True)
            # consider cost rate
            current_tradable_value /= (1 + max(trade_exchange.close_cost, trade_exchange.open_cost))
            # consider risk degree
            current_tradable_value *= risk_degree
            target_amount_dict = trade_exchange.generate_amount_position_from_weight_position(
                weight_position=target_weight_position, cash=current_tradable_value, trade_date=trade_date)

            return target_amount_dict

            pass

        def generate_order_list_from_target_amount_position(self, current, trade_exchange, target_amount_position, trade_date):
            '''Parameter:
            current : Position()
            trade_exchange : Exchange()
            target_amount_position : {stock_id : amount}
            trade_date : trade date
            generate order list from weight_position
            '''
            # strategy：
            current_amount_dict = current.get_stock_amount_dict()
            order_list = trade_exchange.generate_order_for_target_amount_position(target_position=target_amount_position,
                                                                                  current_position=current_amount_dict,
                                                                                  trade_date=trade_date)
            return order_list

        def generate_order_list_from_target_weight_position(self, current, trade_exchange, target_weight_position, risk_degree ,trade_date, interact=True):
            '''
            generate order_list from weight_position
            use API from trade_exchage
            current : Postion(), current position
            trade_exchange : Exchange()
            target_weight_position : {stock_id : weight}
            risk_degree : 0-1, 0.95 for example, use 95% money to trade
            trade_date : trade date
            interact : bool
            :return: order_list
            '''
            # calculate value for current position
            current_amount_dict = current.get_stock_amount_dict()
            current_tradable_value = trade_exchange.calculate_amount_position_value(amount_dict=current_amount_dict, trade_date=trade_date, only_tradable=True)
            # add cash
            current_tradable_value += current.get_cash()
            # consider cost rate
            current_tradable_value /= (1+max(trade_exchange.close_cost, trade_exchange.open_cost))
            # consider risk degree
            current_tradable_value *= risk_degree
            # Strategy 1 : generate amount_position from weight_position
            # use API of trade_exchange
            target_amount_dict = trade_exchange.generate_amount_position_from_weight_position(weight_position=target_weight_position, cash=current_tradable_value, trade_date=trade_date)
            order_list = trade_exchange.generate_order_for_target_amount_position(target_position=target_amount_dict, current_position=current_amount_dict, trade_date=trade_date)

            return order_list

        def generate_order_list(self, score_series, current, trade_exchange, trade_date, topk, margin, risk_degree):
            '''
            score_series: pred score for this trade date, pd.Series, index is stock_id, contain 'score' column
            current: Postion(), current position
            trade_exchange: trade date
            trade_date:
            topk: topk
            margin: buffer margin
            risk_degree: risk_degree : 0-1, 0.95 for example, use 95% money to trade
            :return: order list : list of Order()
            '''
            # generate_order_list
            # strategy 1，generate_target_weight_position() and xecute_target_weight_position_by_order_list() for order_list
            if not self.is_adjust(trade_date):
                return []
            target_weigth_position = self.generate_target_weight_position(score=score_series,
                                                                          current=current,
                                                                          trade_exchange=trade_exchange,
                                                                          topk=topk,
                                                                          buffer_margin=margin,
                                                                          trade_date=trade_date,
                                                                          risk_degree=risk_degree
                                                                          )
            order_list = self.generate_order_list_from_target_weight_positione(    current=current,
                                                                               trade_exchange=trade_exchange,
                                                                               target_weight_position=target_weigth_position,
                                                                               risk_degree=risk_degree,
                                                                               trade_date=trade_date)


            # strategy 2 : amount_position's view generate_target_amount_position() and generate_order_list_from_target_amount_position() to generate order_list
            target_amount_position = self.generate_target_amount_position(score=score_series,
                                                                          current=current,
                                                                          trade_exchange=trade_exchange,
                                                                          target_weight_position=None,
                                                                          topk=topk,
                                                                          buffer_margin=margin,
                                                                          trade_date=trade_date,
                                                                          risk_degree=risk_degree
                                                                          )
            order_list = self.generate_order_list_from_target_amount_position(current=current,
                                                                           trade_exchange=trade_exchange,
                                                                           target_amount_position=target_amount_position,
                                                                           trade_date=trade_date)
            return order_list
    """

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


def get_sell_limit(score, buffer_margin):
    """get_sell_limit

    :param score: pred score for this trade date, pd.Series, index is stock_id, contain 'score' column
    :param buffer_margin: int or float
    """
    if isinstance(buffer_margin, int):
        return buffer_margin
    else:
        if buffer_margin < 0.0 or buffer_margin > 1.0:
            raise ValueError("Buffer margin should range in [0, 1]")
        return int(score.count() * buffer_margin)


class MarginInterface:
    def get_buffer_margin(self, trade_date):
        """get_buffer_margin
        Get the buffer margin dynamically for topk strategy.

        :param trade_date: trading date
        """
        raise NotImplementedError("Please implement the margin dynamically")


class TopkWeightStrategy(ListAdjustTimer, WeightStrategyBase, MarginInterface):
    # NOTE: The list adjust Timer must be placed before WeightStrategyBase before.
    def __init__(self, topk, buffer_margin, risk_degree=0.95, **kwargs):
        """Parameter
        topk : int
            top-N stocks to buy

        buffer_margin : int or float
            if isinstance(margin, int):
                sell_limit = margin
            else:
                sell_limit = pred_in_a_day.count() * margin
            buffer margin, in single score_mode, continue holding stock if it is in nlargest(sell_limit)
            sell_limit should be no less than topk

        risk_degree : float
            position percentage of total value
        """
        WeightStrategyBase.__init__(self, **kwargs)
        ListAdjustTimer.__init__(self, kwargs.get("adjust_dates", None))
        self.topk = topk
        self.buffer_margin = buffer_margin
        self.risk_degree = risk_degree

    def get_risk_degree(self, date):
        """get_risk_degree
        Return the proportion of your total value you will used in investment.
        Dynamically risk_degree will result in Market timing
        """
        # It will use 95% amoutn of your total value by default
        return self.risk_degree

    def get_buffer_margin(self, trade_date):
        return self.buffer_margin

    def generate_target_weight_position(self, score, current, trade_date):
        """Parameter:
        score : pred score for this trade date, pd.Series, index is stock_id, contain 'score' column
        current : current position, use Position() class
        trade_exchange : Exchange()
        trade_date : trade date
        generate target position from score for this date and the current position
        The cache is not considered in the position
        """
        sell_limit = get_sell_limit(score, self.get_buffer_margin(trade_date))
        buffer = score.sort_values(ascending=False).iloc[:sell_limit]
        if sell_limit <= self.topk:
            # no buffer
            target_weight_position = {code: 1 / self.topk for code in buffer.index}
        else:
            # buffer is considered
            current_stock_list = current.get_stock_list()
            mask = buffer.index.isin(current_stock_list)
            keep = set(buffer[mask].index)
            new_stock_list = list(keep)
            if len(keep) < self.topk:
                new_stock_list += list(buffer[~mask].index[: self.topk - len(keep)])
            else:
                # truncate the stocks
                new_stock_list.sort(key=score.get, reverse=True)
                new_stock_list = new_stock_list[: self.topk]
            target_weight_position = {code: 1 / self.topk for code in new_stock_list}
        return target_weight_position


class TopkAmountStrategy(BaseStrategy, MarginInterface, ListAdjustTimer):
    def __init__(self, topk, buffer_margin, risk_degree=0.95, thresh=1, hold_thresh=1, **kwargs):
        """Parameter
        topk : int
            top-N stocks to buy
        buffer_margin : int or float
            if isinstance(margin, int):
                sell_limit = margin
            else:
                sell_limit = pred_in_a_day.count() * margin
            buffer margin, in single score_mode, continue holding stock if it is in nlargest(sell_limit)
            sell_limit should be no less than topk
        risk_degree : float
            position percentage of total value
        thresh : int
            minimun holding days since last buy singal of the stock
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.thresh
        """
        BaseStrategy.__init__(self)
        ListAdjustTimer.__init__(self, kwargs.get("adjust_dates", None))
        self.topk = topk
        self.buffer_margin = buffer_margin
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

    def get_buffer_margin(self, trade_date):
        return self.buffer_margin

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
        # generate order list for this adjust date
        current_temp = copy.deepcopy(
            current
        )  # this copy is necessary. Due to the trade_exchange.calc_deal_order will simulate the dealing process

        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        buffer = score_series.sort_values(ascending=False).iloc[
            : get_sell_limit(score_series, self.get_buffer_margin(trade_date))
        ]
        current_stock_list = current_temp.get_stock_list()
        mask = buffer.index.isin(current_stock_list)
        keep = set(buffer[mask].index)
        # stocks that get buy signals
        buy = set(buffer.iloc[: self.topk].index)
        new = buffer[~mask].index.get_level_values(0)  # new stocks to buy
        # sell stock not in keep
        # sell mode: sell all
        for code in current_stock_list:
            if not trade_exchange.is_stock_tradable(stock_id=code, trade_date=trade_date):
                continue
            if code not in keep:
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
                # Only in margin will no produce buy signal
                self.stock_count[code] = 1
            else:
                self.stock_count[code] += 1
        # buy new stock
        # note the current has been changed
        current_stock_list = current_temp.get_stock_list()
        n_buy = self.topk - len(current_stock_list)
        value = cash * self.risk_degree / n_buy if n_buy > 0 else 0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not consider it
        # as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+trade_exchange.open_cost) # set open_cost limit
        for code in new[:n_buy]:
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


class TopkDropoutStrategy(BaseStrategy, ListAdjustTimer):
    def __init__(self, topk, n_drop, method="bottom", risk_degree=0.95, thresh=1, hold_thresh=1, **kwargs):
        """Parameter
        topk : int
            top-N stocks to buy
        n_drop : int
            number of stocks to be replaced
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
                    # excute the order
                    trade_val, trade_cost, trade_price = trade_exchange.calc_deal_order(sell_order)
                    # update cash
                    cash += trade_val - trade_cost
                    # updte current
                    current_temp.update_order(sell_order, trade_price)
                    # sold
                    del self.stock_count[code]
                else:
                    # no buy signal, but the stock is kept
                    self.stock_count[code] += 1
            elif code in buy:
                # NOTE: This is different from the original version
                # get new buy signal
                # Only the stock fall in to topk will produce buy signal
                # Only in margin will no produce buy signal
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
