"""Signal-driven strategies including LongShortTopKStrategy (crypto-ready)."""

# pylint: disable=C0301,R0912,R0915,R0902,R0913,R0914,C0411,W0511,W0718,W0612,W0613,C0209,W1309,C1802,C0115,C0116
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import copy
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List, Text, Tuple, Union, Optional
from abc import ABC

from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.log import get_module_logger
from qlib.utils import get_pre_trading_date, load_dataset
from qlib.contrib.strategy.order_generator import OrderGenerator, OrderGenWOInteract
from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer


class BaseSignalStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame] = None,
        model=None,
        dataset=None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        risk_degree : float
            position percentage of total value.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super().__init__(level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs)

        self.risk_degree = risk_degree

        # This is trying to be compatible with previous version of qlib task config
        if model is not None and dataset is not None:
            warnings.warn("`model` `dataset` is deprecated; use `signal`.", DeprecationWarning)
            signal = model, dataset

        self.signal: Signal = create_signal_from(signal)

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will use in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amount of your total value by default
        return self.risk_degree


class TopkDropoutStrategy(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    # 4. Regenerate results with forbid_all_trade_at_limit set to false and flip the default to false, as it is consistent with reality.
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
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
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.

            if only_tradable:

                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.

            else:

                strategy will make buy sell decision without checking the tradable state of the stock.
        forbid_all_trade_at_limit : bool
            if forbid all trades when limit_up or limit_down reached.

            if forbid_all_trade_at_limit:

                strategy will not do any trade when price reaches limit up/down, even not sell at limit up nor buy at
                limit down, though allowed in reality.

            else:

                strategy will sell at limit up and buy ad limit down.
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)
        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:
            # Otherwise, the stock will make decision without the stock tradable info
            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        current_temp: Position = copy.deepcopy(self.trade_position)
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
            except ValueError:  # No enough candidates
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # Get the stock list we really want to buy
        buy = today[: len(sell) + self.topk - len(last)]
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                # check hold limit
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                # sell order
                sell_amount = current_temp.get_stock_amount(code=code)
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
        # current_stock_list = current_temp.get_stock_list()
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
        # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock suspended
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
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


class WeightStrategyBase(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        *,
        order_generator_cls_or_obj=OrderGenWOInteract,
        **kwargs,
    ):
        """
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report

            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:

                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.
        """
        super().__init__(**kwargs)

        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator: OrderGenerator = order_generator_cls_or_obj()
        else:
            self.order_generator: OrderGenerator = order_generator_cls_or_obj

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """
        Generate target position from score for this date and the current position.The cash is not considered in the position

        Parameters
        -----------
        score : pd.Series
            pred score for this trade date, index is stock_id, contain 'score' column.
        current : Position()
            current position.
        trade_start_time: pd.Timestamp
        trade_end_time: pd.Timestamp
        """
        raise NotImplementedError()

    def generate_trade_decision(self, execute_result=None):
        # generate_trade_decision
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
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


class EnhancedIndexingStrategy(WeightStrategyBase):
    """Enhanced Indexing Strategy

    Enhanced indexing combines the arts of active management and passive management,
    with the aim of outperforming a benchmark index (e.g., S&P 500) in terms of
    portfolio return while controlling the risk exposure (a.k.a. tracking error).

    Users need to prepare their risk model data like below:

    .. code-block:: text

        ├── /path/to/riskmodel
        ├──── 20210101
        ├────── factor_exp.{csv|pkl|h5}
        ├────── factor_cov.{csv|pkl|h5}
        ├────── specific_risk.{csv|pkl|h5}
        ├────── blacklist.{csv|pkl|h5}  # optional

    The risk model data can be obtained from risk data provider. You can also use
    `qlib.model.riskmodel.structured.StructuredCovEstimator` to prepare these data.

    Args:
        riskmodel_path (str): risk model path
        name_mapping (dict): alternative file names
    """

    FACTOR_EXP_NAME = "factor_exp.pkl"
    FACTOR_COV_NAME = "factor_cov.pkl"
    SPECIFIC_RISK_NAME = "specific_risk.pkl"
    BLACKLIST_NAME = "blacklist.pkl"

    def __init__(
        self,
        *,
        riskmodel_root,
        market="csi500",
        turn_limit=None,
        name_mapping=None,
        optimizer_kwargs=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logger = get_module_logger("EnhancedIndexingStrategy")

        self.riskmodel_root = riskmodel_root
        self.market = market
        self.turn_limit = turn_limit

        name_mapping = {} if name_mapping is None else name_mapping
        self.factor_exp_path = name_mapping.get("factor_exp", self.FACTOR_EXP_NAME)
        self.factor_cov_path = name_mapping.get("factor_cov", self.FACTOR_COV_NAME)
        self.specific_risk_path = name_mapping.get("specific_risk", self.SPECIFIC_RISK_NAME)
        self.blacklist_path = name_mapping.get("blacklist", self.BLACKLIST_NAME)

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.optimizer = EnhancedIndexingOptimizer(**optimizer_kwargs)

        self.verbose = verbose

        self._riskdata_cache = {}

    def get_risk_data(self, date):
        if date in self._riskdata_cache:
            return self._riskdata_cache[date]

        root = self.riskmodel_root + "/" + date.strftime("%Y%m%d")
        if not os.path.exists(root):
            return None

        factor_exp = load_dataset(root + "/" + self.factor_exp_path, index_col=[0])
        factor_cov = load_dataset(root + "/" + self.factor_cov_path, index_col=[0])
        specific_risk = load_dataset(root + "/" + self.specific_risk_path, index_col=[0])

        if not factor_exp.index.equals(specific_risk.index):
            # NOTE: for stocks missing specific_risk, we always assume it has the highest volatility
            specific_risk = specific_risk.reindex(factor_exp.index, fill_value=specific_risk.max())

        universe = factor_exp.index.tolist()

        blacklist = []
        if os.path.exists(root + "/" + self.blacklist_path):
            blacklist = load_dataset(root + "/" + self.blacklist_path).index.tolist()

        self._riskdata_cache[date] = factor_exp.values, factor_cov.values, specific_risk.values, universe, blacklist

        return self._riskdata_cache[date]

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        trade_date = trade_start_time
        pre_date = get_pre_trading_date(trade_date, future=True)  # previous trade date

        # load risk data
        outs = self.get_risk_data(pre_date)
        if outs is None:
            self.logger.warning(f"no risk data for {pre_date:%Y-%m-%d}, skip optimization")
            return None
        factor_exp, factor_cov, specific_risk, universe, blacklist = outs

        # transform score
        # NOTE: for stocks missing score, we always assume they have the lowest score
        score = score.reindex(universe).fillna(score.min()).values

        # get current weight
        # NOTE: if a stock is not in universe, its current weight will be zero
        cur_weight = current.get_stock_weight_dict(only_stock=False)
        cur_weight = np.array([cur_weight.get(stock, 0) for stock in universe])
        assert all(cur_weight >= 0), "current weight has negative values"
        cur_weight = cur_weight / self.get_risk_degree(trade_date)  # sum of weight should be risk_degree
        if cur_weight.sum() > 1 and self.verbose:
            self.logger.warning(f"previous total holdings excess risk degree (current: {cur_weight.sum()})")

        # load bench weight
        bench_weight = D.features(
            D.instruments("all"), [f"${self.market}_weight"], start_time=pre_date, end_time=pre_date
        ).squeeze()
        bench_weight.index = bench_weight.index.droplevel(level="datetime")
        bench_weight = bench_weight.reindex(universe).fillna(0).values

        # whether stock tradable
        # NOTE: currently we use last day volume to check whether tradable
        tradable = D.features(D.instruments("all"), ["$volume"], start_time=pre_date, end_time=pre_date).squeeze()
        tradable.index = tradable.index.droplevel(level="datetime")
        tradable = tradable.reindex(universe).gt(0).values
        mask_force_hold = ~tradable

        # mask force sell
        mask_force_sell = np.array([stock in blacklist for stock in universe], dtype=bool)

        # optimize
        weight = self.optimizer(
            r=score,
            F=factor_exp,
            cov_b=factor_cov,
            var_u=specific_risk**2,
            w0=cur_weight,
            wb=bench_weight,
            mfh=mask_force_hold,
            mfs=mask_force_sell,
        )

        target_weight_position = {stock: weight for stock, weight in zip(universe, weight) if weight > 0}

        if self.verbose:
            self.logger.info("trade date: {:%Y-%m-%d}".format(trade_date))
            self.logger.info("number of holding stocks: {}".format(len(target_weight_position)))
            self.logger.info("total holding weight: {:.6f}".format(weight.sum()))

        return target_weight_position


class LongShortTopKStrategy(BaseSignalStrategy):
    """
    Strict TopK-aligned Long-Short strategy.

    - Uses shift=1 signals (previous bar's signal for current trading) like TopkDropoutStrategy
    - Maintains separate TopK pools for long and short legs with independent rotation (n_drop)
    - Respects tradability checks and limit rules consistent with TopkDropoutStrategy
    - Requires a shortable exchange to open short positions; otherwise SELL will be clipped by Exchange
    """

    def __init__(
        self,
        *,
        topk_long: int,
        topk_short: int,
        n_drop_long: int,
        n_drop_short: int,
        method_sell: str = "bottom",
        method_buy: str = "top",
        hold_thresh: int = 1,
        only_tradable: bool = False,
        forbid_all_trade_at_limit: bool = True,
        rebalance_to_weights: bool = True,
        long_share: Optional[float] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.topk_long = topk_long
        self.topk_short = topk_short
        self.n_drop_long = n_drop_long
        self.n_drop_short = n_drop_short
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        self.rebalance_to_weights = rebalance_to_weights
        # When both legs enabled, split risk_degree by long_share (0~1). None -> 0.5 default.
        self.long_share = long_share
        self._debug = debug

    def generate_trade_decision(self, execute_result=None):
        # Align time windows (shift=1)
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        # Helper functions copied from TopkDropoutStrategy semantics
        if self.only_tradable:

            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:

            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        current_temp: Position = copy.deepcopy(self.trade_position)

        # Build current long/short lists by sign of amount
        current_stock_list = current_temp.get_stock_list()
        long_now = []  # amounts > 0
        short_now = []  # amounts < 0
        for code in current_stock_list:
            amt = current_temp.get_stock_amount(code)
            if amt > 0:
                long_now.append(code)
            elif amt < 0:
                short_now.append(code)
        if self._debug:
            print(
                f"[LongShortTopKStrategy][{trade_start_time}] init_pos: longs={len(long_now)}, shorts={len(short_now)}"
            )
            if short_now:
                try:
                    details = [(c, float(current_temp.get_stock_amount(c))) for c in short_now]
                    print(f"[LongShortTopKStrategy][{trade_start_time}] short_details: {details}")
                except Exception as e:
                    get_module_logger("LongShortTopKStrategy").debug("Failed to print short_details: %s", e)

        # ---- Long leg selection (descending score) ----
        last_long = pred_score.reindex(long_now).sort_values(ascending=False).index
        n_to_add_long = max(0, self.n_drop_long + self.topk_long - len(last_long))
        if self.method_buy == "top":
            today_long_candi = get_first_n(
                pred_score[~pred_score.index.isin(last_long)].sort_values(ascending=False).index,
                n_to_add_long,
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk_long)
            candi = list(filter(lambda x: x not in last_long, topk_candi))
            try:
                today_long_candi = (
                    list(np.random.choice(candi, n_to_add_long, replace=False)) if n_to_add_long > 0 else []
                )
            except ValueError:
                today_long_candi = candi
        else:
            raise NotImplementedError
        comb_long = pred_score.reindex(last_long.union(pd.Index(today_long_candi))).sort_values(ascending=False).index
        if self.method_sell == "bottom":
            sell_long = last_long[last_long.isin(get_last_n(comb_long, self.n_drop_long))]
        elif self.method_sell == "random":
            candi = filter_stock(last_long)
            try:
                sell_long = pd.Index(np.random.choice(candi, self.n_drop_long, replace=False) if len(candi) else [])
            except ValueError:
                sell_long = pd.Index(candi)
        else:
            raise NotImplementedError
        buy_long = today_long_candi[: len(sell_long) + self.topk_long - len(last_long)]

        # ---- Short leg selection (ascending score) ----
        last_short = pred_score.reindex(short_now).sort_values(ascending=True).index
        n_to_add_short = max(0, self.n_drop_short + self.topk_short - len(last_short))
        if self.method_buy == "top":  # for short, "top" means most negative i.e., ascending
            today_short_candi = get_first_n(
                pred_score[~pred_score.index.isin(last_short)].sort_values(ascending=True).index,
                n_to_add_short,
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=True).index, self.topk_short)
            candi = list(filter(lambda x: x not in last_short, topk_candi))
            try:
                today_short_candi = (
                    list(np.random.choice(candi, n_to_add_short, replace=False)) if n_to_add_short > 0 else []
                )
            except ValueError:
                today_short_candi = candi
        else:
            raise NotImplementedError
        comb_short = pred_score.reindex(last_short.union(pd.Index(today_short_candi))).sort_values(ascending=True).index
        if self.method_sell == "bottom":  # for short, bottom means highest scores among shorts (least negative)
            cover_short = last_short[last_short.isin(get_last_n(comb_short, self.n_drop_short))]
        elif self.method_sell == "random":
            candi = filter_stock(last_short)
            try:
                cover_short = pd.Index(np.random.choice(candi, self.n_drop_short, replace=False) if len(candi) else [])
            except ValueError:
                cover_short = pd.Index(candi)
        else:
            raise NotImplementedError
        open_short = today_short_candi[: len(cover_short) + self.topk_short - len(last_short)]

        # ---- Rebalance to target weights to bound gross leverage and net exposure ----
        # Determine final long/short sets considering hold_thresh and tradability
        def can_trade(code: str, direction: int) -> bool:
            return self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else direction,
            )

        time_per_step = self.trade_calendar.get_freq()

        # apply hold_thresh when removing
        actual_sold_longs = set()
        for code in last_long:
            if (
                code in sell_long
                and current_temp.get_stock_count(code, bar=time_per_step) >= self.hold_thresh
                and can_trade(code, OrderDir.SELL)
            ):
                actual_sold_longs.add(code)

        actual_covered_shorts = set()
        # Align with TopK: in long-only mode, fully cover any existing shorts (not limited by n_drop_short or hold_thresh)
        long_only_mode = (self.topk_short is None) or (self.topk_short <= 0)
        if long_only_mode:
            # Only cover when there is a real negative position to avoid false positives
            for code in last_short:
                if current_temp.get_stock_amount(code) < 0 and can_trade(code, OrderDir.BUY):
                    actual_covered_shorts.add(code)
        else:
            for code in last_short:
                if (
                    code in cover_short
                    and current_temp.get_stock_count(code, bar=time_per_step) >= self.hold_thresh
                    and can_trade(code, OrderDir.BUY)
                ):
                    actual_covered_shorts.add(code)
        if self._debug:
            print(
                f"[LongShortTopKStrategy][{trade_start_time}] cover_shorts={len(actual_covered_shorts)} buy_longs_plan={len(buy_long)} open_shorts_plan={len(open_short)}"
            )

        # Preserve raw planned lists before tradability filtering to align with TopK semantics
        raw_buy_long = list(buy_long)
        raw_open_short = list(open_short)

        buy_long = [c for c in buy_long if can_trade(c, OrderDir.BUY)]
        open_short = [c for c in open_short if can_trade(c, OrderDir.SELL)]
        open_short = [c for c in open_short if c not in buy_long]  # avoid overlap

        final_long_set = (set(long_now) - actual_sold_longs) | set(buy_long)
        final_short_set = (set(short_now) - actual_covered_shorts) | set(open_short)

        # Optional: TopK-style no-rebalance branch (symmetric long/short)
        if not self.rebalance_to_weights:
            order_list: List[Order] = []
            cash = current_temp.get_cash()

            # 1) Sell dropped longs entirely
            for code in long_now:
                if code in actual_sold_longs and can_trade(code, OrderDir.SELL):
                    sell_amount = current_temp.get_stock_amount(code=code)
                    if sell_amount <= 0:
                        continue
                    sell_order = Order(
                        stock_id=code,
                        amount=sell_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,
                    )
                    if self.trade_exchange.check_order(sell_order):
                        order_list.append(sell_order)
                        trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                            sell_order, position=current_temp
                        )
                        cash += trade_val - trade_cost

            # Snapshot cash AFTER long sells but BEFORE short covers
            # TopK-style long leg should allocate based on this snapshot to avoid
            # short-cover cash consumption leaking into long-buy budget.
            cash_after_long_sells = cash

            # 2) Cover dropped shorts entirely (BUY to cover)
            for code in short_now:
                if code in actual_covered_shorts and can_trade(code, OrderDir.BUY):
                    cover_amount = abs(current_temp.get_stock_amount(code=code))
                    if cover_amount <= 0:
                        continue
                    cover_order = Order(
                        stock_id=code,
                        amount=cover_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.BUY,
                    )
                    if self.trade_exchange.check_order(cover_order):
                        order_list.append(cover_order)
                        trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                            cover_order, position=current_temp
                        )
                        cash -= trade_val + trade_cost  # covering consumes cash

            # 3) Buy new longs with equal cash split, honoring risk_degree
            rd = float(self.get_risk_degree(trade_step))
            # Allocate long/short share: support long_share; degenerate for single-leg mode
            short_only_mode = (self.topk_long is None) or (self.topk_long <= 0)
            share = self.long_share if (self.long_share is not None) else 0.5
            if long_only_mode:
                rd_long, rd_short = rd, 0.0
            elif short_only_mode:
                rd_long, rd_short = 0.0, rd
            else:
                rd_long, rd_short = rd * share, rd * (1.0 - share)
            if self._debug:
                print(
                    f"[LongShortTopKStrategy][{trade_start_time}] rd={rd:.4f} rd_long={rd_long:.4f} rd_short={rd_short:.4f} cash_after_long_sells={cash_after_long_sells:.2f}"
                )
            # Align with TopK: use cash snapshot after long sells; split by planned count (raw)
            value_per_buy = cash_after_long_sells * rd_long / len(raw_buy_long) if len(raw_buy_long) > 0 else 0.0
            for code in raw_buy_long:
                if not can_trade(code, OrderDir.BUY):
                    continue
                price = self.trade_exchange.get_deal_price(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
                )
                if price is None or not np.isfinite(price) or price <= 0:
                    continue
                buy_amount = value_per_buy / float(price)
                factor = self.trade_exchange.get_factor(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time
                )
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
                if buy_amount <= 0:
                    continue
                buy_order = Order(
                    stock_id=code,
                    amount=buy_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                )
                order_list.append(buy_order)

            # 4) Open new shorts equally by target short notional derived from rd
            # Compute current short notional after covering
            def _get_price(sid: str, direction: int):
                px = self.trade_exchange.get_deal_price(
                    stock_id=sid, start_time=trade_start_time, end_time=trade_end_time, direction=direction
                )
                return float(px) if (px is not None and np.isfinite(px) and px > 0) else None

            # Recompute equity after previous simulated deals
            # For TopK parity, compute equity BEFORE executing new long buys and BEFORE opening new shorts
            # i.e., after simulated sells/covers above.
            equity = max(1e-12, float(current_temp.calculate_value()))

            # Sum current short notional
            current_short_value = 0.0
            for sid in current_temp.get_stock_list():
                amt = current_temp.get_stock_amount(sid)
                if amt < 0:
                    px = _get_price(sid, OrderDir.BUY)  # price to cover
                    if px is not None:
                        current_short_value += abs(float(amt)) * px

            # Use the same rd_short allocation as above
            # Note: if short_only_mode, rd_long = 0 and rd_short = rd
            # Reuse the rd_short computed earlier
            desired_short_value = equity * rd_short
            remaining_short_value = max(0.0, desired_short_value - current_short_value)
            # Align with TopK: split by planned short-open count (raw), then check tradability
            value_per_short_open = remaining_short_value / len(raw_open_short) if len(raw_open_short) > 0 else 0.0
            if self._debug:
                print(
                    f"[LongShortTopKStrategy][{trade_start_time}] equity={equity:.2f} cur_short_val={current_short_value:.2f} desired_short_val={desired_short_value:.2f} rem_short_val={remaining_short_value:.2f} v_per_short={value_per_short_open:.2f}"
                )

            for code in raw_open_short:
                if not can_trade(code, OrderDir.SELL):
                    continue
                price = _get_price(code, OrderDir.SELL)
                if price is None:
                    continue
                sell_amount = value_per_short_open / float(price)
                factor = self.trade_exchange.get_factor(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time
                )
                sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
                if sell_amount <= 0:
                    continue
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
                order_list.append(sell_order)

            return TradeDecisionWO(order_list, self)

        # Target weights
        rd = float(self.get_risk_degree(trade_step))
        share = self.long_share if (self.long_share is not None) else 0.5
        long_total = 0.0
        short_total = 0.0
        if len(final_long_set) > 0 and len(final_short_set) > 0:
            long_total = rd * share
            short_total = rd * (1.0 - share)
        elif len(final_long_set) > 0:
            long_total = rd
        elif len(final_short_set) > 0:
            short_total = rd

        target_weight: Dict[str, float] = {}
        if len(final_long_set) > 0:
            lw = long_total / len(final_long_set)
            for c in final_long_set:
                target_weight[c] = lw
        if len(final_short_set) > 0:
            sw = -short_total / len(final_short_set)
            for c in final_short_set:
                target_weight[c] = sw

        # Stocks to liquidate
        for c in current_temp.get_stock_list():
            if c not in target_weight:
                target_weight[c] = 0.0

        # Generate orders by comparing current vs target
        order_list: List[Order] = []
        equity = max(1e-12, float(current_temp.calculate_value()))
        for code, tw in target_weight.items():
            # get price
            # We select direction by desired delta later, here just fetch a price using BUY as placeholder if needed
            price_buy = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            price_sell = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL
            )
            price = price_buy if price_buy else price_sell
            if not price or price <= 0:
                continue
            cur_amount = float(current_temp.get_stock_amount(code)) if code in current_temp.get_stock_list() else 0.0
            cur_value = cur_amount * price
            tgt_value = tw * equity
            delta_value = tgt_value - cur_value
            if abs(delta_value) <= 0:
                continue
            direction = OrderDir.BUY if delta_value > 0 else OrderDir.SELL
            if not can_trade(code, direction):
                continue
            delta_amount = abs(delta_value) / price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            delta_amount = self.trade_exchange.round_amount_by_trade_unit(delta_amount, factor)
            if delta_amount <= 0:
                continue
            order_list.append(
                Order(
                    stock_id=code,
                    amount=delta_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=direction,
                )
            )

        return TradeDecisionWO(order_list, self)
