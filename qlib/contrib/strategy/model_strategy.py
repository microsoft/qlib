import copy
from qlib.backtest.position import Position
import warnings
import numpy as np
import pandas as pd

from ...utils.resam import resam_ts_data
from ...strategy.base import ModelStrategy
from ...backtest.decision import Order, BaseTradeDecision, OrderDir, TradeDecisionWO

from .order_generator import OrderGenWInteract


class TopkDropoutStrategy(ModelStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        model,
        dataset,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        risk_degree=0.95,
        hold_thresh=1,
        only_tradable=False,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
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
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.
            if only_tradable:
                strategy will make buy sell decision without checking the tradable state of the stock.
            else:
                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it run faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super(TopkDropoutStrategy, self).__init__(
            model, dataset, level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs
        )
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.risk_degree = risk_degree
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will used in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amoutn of your total value by default
        return self.risk_degree

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = resam_ts_data(self.pred_scores, start_time=pred_start_time, end_time=pred_end_time, method="last")
        if pred_score is None:
            return TradeDecisionWO([], self)
        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            def get_first_n(l, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(l) if reverse else l:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(l, n):
                return get_first_n(l, n, reverse=True)

            def filter_stock(l):
                return [
                    si
                    for si in l
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:
            # Otherwise, the stock will make decision with out the stock tradable info
            def get_first_n(l, n):
                return list(l)[:n]

            def get_last_n(l, n):
                return list(l)[-n:]

            def filter_stock(l):
                return l

        current_temp = copy.deepcopy(self.trade_position)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        # last position (sorted by score)
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
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
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

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
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            if code in sell:
                # check hold limit
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                # sell order
                sell_amount = current_temp.get_stock_amount(code=code)
                factor = self.trade_exchange.get_factor(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time
                )
                # sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,  # 0 for sell, 1 for buy
                )
                # is order executable
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    # update cash
                    cash += trade_val - trade_cost
        # buy new stock
        # note the current has been changed
        current_stock_list = current_temp.get_stock_list()
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
        # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock suspended
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            # buy order
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,  # 1 for buy
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)


class WeightStrategyBase(ModelStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        model,
        dataset,
        order_generator_cls_or_obj=OrderGenWInteract,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it run faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.
        """
        super(WeightStrategyBase, self).__init__(
            model, dataset, level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs
        )
        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator = order_generator_cls_or_obj()
        else:
            self.order_generator = order_generator_cls_or_obj

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will used in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amoutn of your total value by default
        return 0.95

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
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

    def generate_trade_decision(self, execute_result=None):
        # generate_trade_decision
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = resam_ts_data(self.pred_scores, start_time=pred_start_time, end_time=pred_end_time, method="last")
        if pred_score is None:
            return TradeDecisionWO([], self)
        current_temp = copy.deepcopy(self.trade_position)
        assert isinstance(current_temp, Position)  # Avoid InfPosition

        target_weight_position = self.generate_target_weight_position(
            score=pred_score, current=current_temp, trade_start_time=trade_start_time, trade_end_time=trade_end_time
        )
        order_list = self.order_generator.generate_order_list_from_target_weight_position(
            current=current_temp,
            trade_exchange=self.trade_exchange,
            risk_degree=self.get_risk_degree(trade_step),
            target_weight_position=target_weight_position,
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        return TradeDecisionWO(order_list, self)
