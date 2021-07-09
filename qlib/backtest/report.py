# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections import OrderedDict
from logging import warning
import pathlib
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from pandas.core import groupby
from pandas.core.frame import DataFrame

from qlib.backtest.exchange import Exchange
from qlib.backtest.order import BaseTradeDecision, Order, OrderDir
from qlib.backtest.utils import TradeCalendarManager

from ..data import D
from ..tests.config import CSI300_BENCH
from ..utils.resam import get_higher_eq_freq_feature, resam_ts_data
from ..utils.time import Freq


class Report:
    """
    Motivation:
        Report is for supporting portfolio related metrics.

    Implementation:
        daily report of the account
        contain those followings: returns, costs turnovers, accounts, cash, bench, value
        update report
    """

    def __init__(self, freq: str = "day", benchmark_config: dict = {}):
        """
        Parameters
        ----------
        freq : str
            frequency of trading bar, used for updating hold count of trading bar
        benchmark_config : dict
            config of benchmark, may including the following arguments:
            - benchmark : Union[str, list, pd.Series]
                - If `benchmark` is pd.Series, `index` is trading date; the value T is the change from T-1 to T.
                    example:
                        print(D.features(D.instruments('csi500'), ['$close/Ref($close, 1)-1'])['$close/Ref($close, 1)-1'].head())
                            2017-01-04    0.011693
                            2017-01-05    0.000721
                            2017-01-06   -0.004322
                            2017-01-09    0.006874
                            2017-01-10   -0.003350
                - If `benchmark` is list, will use the daily average change of the stock pool in the list as the 'bench'.
                - If `benchmark` is str, will use the daily change as the 'bench'.
                benchmark code, default is SH000300 CSI300
            - start_time : Union[str, pd.Timestamp], optional
                - If `benchmark` is pd.Series, it will be ignored
                - Else, it represent start time of benchmark, by default None
            - end_time : Union[str, pd.Timestamp], optional
                - If `benchmark` is pd.Series, it will be ignored
                - Else, it represent end time of benchmark, by default None

        """
        self.init_vars()
        self.init_bench(freq=freq, benchmark_config=benchmark_config)

    def init_vars(self):
        self.accounts = OrderedDict()  # account postion value for each trade time
        self.returns = OrderedDict()  # daily return rate for each trade time
        self.total_turnovers = OrderedDict()  # total turnover for each trade time
        self.turnovers = OrderedDict()  # turnover for each trade time
        self.total_costs = OrderedDict()  # total trade cost for each trade time
        self.costs = OrderedDict()  # trade cost rate for each trade time
        self.values = OrderedDict()  # value for each trade time
        self.cashes = OrderedDict()
        self.benches = OrderedDict()
        self.latest_report_time = None  # pd.TimeStamp

    def init_bench(self, freq=None, benchmark_config=None):
        if freq is not None:
            self.freq = freq
        if benchmark_config is not None:
            self.benchmark_config = benchmark_config
        self.bench = self._cal_benchmark(self.benchmark_config, self.freq)

    def _cal_benchmark(self, benchmark_config, freq):
        benchmark = benchmark_config.get("benchmark", CSI300_BENCH)
        if benchmark is None:
            return None

        if isinstance(benchmark, pd.Series):
            return benchmark
        else:
            start_time = benchmark_config.get("start_time", None)
            end_time = benchmark_config.get("end_time", None)

            if freq is None:
                raise ValueError("benchmark freq can't be None!")
            _codes = benchmark if isinstance(benchmark, (list, dict)) else [benchmark]
            fields = ["$close/Ref($close,1)-1"]
            _temp_result, _ = get_higher_eq_freq_feature(_codes, fields, start_time, end_time, freq=freq)
            if len(_temp_result) == 0:
                raise ValueError(f"The benchmark {_codes} does not exist. Please provide the right benchmark")
            return _temp_result.groupby(level="datetime")[_temp_result.columns.tolist()[0]].mean().fillna(0)

    def _sample_benchmark(self, bench, trade_start_time, trade_end_time):
        if self.bench is None:
            return None

        def cal_change(x):
            return (x + 1).prod()

        _ret = resam_ts_data(bench, trade_start_time, trade_end_time, method=cal_change)
        return 0.0 if _ret is None else _ret - 1

    def is_empty(self):
        return len(self.accounts) == 0

    def get_latest_date(self):
        return self.latest_report_time

    def get_latest_account_value(self):
        return self.accounts[self.latest_report_time]

    def get_latest_total_cost(self):
        return self.total_costs[self.latest_report_time]

    def get_latest_total_turnover(self):
        return self.total_turnovers[self.latest_report_time]

    def update_report_record(
        self,
        trade_start_time=None,
        trade_end_time=None,
        account_value=None,
        cash=None,
        return_rate=None,
        total_turnover=None,
        turnover_rate=None,
        total_cost=None,
        cost_rate=None,
        stock_value=None,
        bench_value=None,
    ):
        # check data
        if None in [
            trade_start_time,
            account_value,
            cash,
            return_rate,
            total_turnover,
            turnover_rate,
            total_cost,
            cost_rate,
            stock_value,
        ]:
            raise ValueError(
                "None in [trade_start_time, account_value, cash, return_rate, total_turnover, turnover_rate, total_cost, cost_rate, stock_value]"
            )

        if trade_end_time is None and bench_value is None:
            raise ValueError("Both trade_end_time and bench_value is None, benchmark is not usable.")
        elif bench_value is None:
            bench_value = self._sample_benchmark(self.bench, trade_start_time, trade_end_time)

        # update report data
        self.accounts[trade_start_time] = account_value
        self.returns[trade_start_time] = return_rate
        self.total_turnovers[trade_start_time] = total_turnover
        self.turnovers[trade_start_time] = turnover_rate
        self.total_costs[trade_start_time] = total_cost
        self.costs[trade_start_time] = cost_rate
        self.values[trade_start_time] = stock_value
        self.cashes[trade_start_time] = cash
        self.benches[trade_start_time] = bench_value
        # update latest_report_date
        self.latest_report_time = trade_start_time
        # finish report update in each step

    def generate_report_dataframe(self):
        report = pd.DataFrame()
        report["account"] = pd.Series(self.accounts)
        report["return"] = pd.Series(self.returns)
        report["total_turnover"] = pd.Series(self.total_turnovers)
        report["turnover"] = pd.Series(self.turnovers)
        report["total_cost"] = pd.Series(self.total_costs)
        report["cost"] = pd.Series(self.costs)
        report["value"] = pd.Series(self.values)
        report["cash"] = pd.Series(self.cashes)
        report["bench"] = pd.Series(self.benches)
        report.index.name = "datetime"
        return report

    def save_report(self, path):
        r = self.generate_report_dataframe()
        r.to_csv(path)

    def load_report(self, path):
        """load report from a file
        should have format like
        columns = ['account', 'return', 'total_turnover', 'turnover', 'cost', 'total_cost', 'value', 'cash', 'bench']
            :param
                path: str/ pathlib.Path()
        """
        path = pathlib.Path(path)
        r = pd.read_csv(open(path, "rb"), index_col=0)
        r.index = pd.DatetimeIndex(r.index)

        index = r.index
        self.init_vars()
        for trade_start_time in index:
            self.update_report_record(
                trade_start_time=trade_start_time,
                account_value=r.loc[trade_start_time]["account"],
                cash=r.loc[trade_start_time]["cash"],
                return_rate=r.loc[trade_start_time]["return"],
                total_turnover=r.loc[trade_start_time]["total_turnover"],
                turnover_rate=r.loc[trade_start_time]["turnover"],
                total_cost=r.loc[trade_start_time]["total_cost"],
                cost_rate=r.loc[trade_start_time]["cost"],
                stock_value=r.loc[trade_start_time]["value"],
                bench_value=r.loc[trade_start_time]["bench"],
            )


class Indicator:
    """
    `Indicator` is implemented in a aggregate way.
    All the metrics are calculated aggregately.
    All the metrics are calculated for a seperated stock and in a specific step on a specific level.

    | indicator    | desc.                                                        |
    |--------------+--------------------------------------------------------------|
    | amount       | the *target* amount given by the outer strategy              |
    | inner_amount | the total *target* amount of inner strategy                  |
    | trade_price  | the average deal price                                       |
    | trade_value  | the total trade value                                        |
    | trade_cost   | the total trade cost  (base price need drection)             |
    | trade_dir    | the trading direction                                        |
    | ffr          | full fill rate                                               |
    | pa           | price advantage                                              |
    | pos          | win rate                                                     |
    | base_price   | the price of baseline                                        |
    | base_volume  | the volume of baseline (for weighted aggregating base_price) |

    **NOTE**:
    The `base_price` and `base_volume` can't be NaN when there are not trading on that step. Otherwise
    aggregating get wrong results.

    So `base_price` will not be calculated in a aggregate way!!

    """

    def __init__(self):
        # order indicator is metrics for a single order for a specific step
        self.order_indicator_his = OrderedDict()
        self.order_indicator: Dict[str, pd.Series] = OrderedDict()

        # trade indicator is metrics for all orders for a specific step
        self.trade_indicator_his = OrderedDict()
        self.trade_indicator: Dict[str, float] = OrderedDict()

        self._trade_calendar = None

    # def reset(self, trade_calendar: TradeCalendarManager):
    def reset(self):
        self.order_indicator = OrderedDict()
        self.trade_indicator = OrderedDict()
        # self._trade_calendar = trade_calendar

    def record(self, trade_start_time):
        self.order_indicator_his[trade_start_time] = self.order_indicator
        self.trade_indicator_his[trade_start_time] = self.trade_indicator

    def _update_order_trade_info(self, trade_info: list):
        amount = dict()
        deal_amount = dict()
        trade_price = dict()
        trade_value = dict()
        trade_cost = dict()
        trade_dir = dict()

        for order, _trade_val, _trade_cost, _trade_price in trade_info:
            amount[order.stock_id] = order.amount_delta
            deal_amount[order.stock_id] = order.deal_amount_delta
            trade_price[order.stock_id] = _trade_price
            trade_value[order.stock_id] = _trade_val * order.sign
            trade_cost[order.stock_id] = _trade_cost
            trade_dir[order.stock_id] = order.direction

        self.order_indicator["amount"] = self.order_indicator["inner_amount"] = pd.Series(amount)
        self.order_indicator["deal_amount"] = pd.Series(deal_amount)
        # NOTE: trade_price and baseline price will be same on the lowest-level
        self.order_indicator["trade_price"] = pd.Series(trade_price)
        self.order_indicator["trade_value"] = pd.Series(trade_value)
        self.order_indicator["trade_cost"] = pd.Series(trade_cost)
        self.order_indicator["trade_dir"] = pd.Series(trade_dir)

    def _update_order_fulfill_rate(self):
        self.order_indicator["ffr"] = self.order_indicator["deal_amount"] / self.order_indicator["amount"]

    def _update_order_price_advantage(self):
        # NOTE:
        # trade_price and baseline price will be same on the lowest-level
        # So Pa should be 0 or do nothing
        self.order_indicator["pa"] = 0

    def update_order_indicators(self, trade_info: list):
        self._update_order_trade_info(trade_info=trade_info)
        self._update_order_fulfill_rate()
        self._update_order_price_advantage()

    def _agg_order_trade_info(self, inner_order_indicators: List[Dict[str, pd.Series]]):
        inner_amount = pd.Series()
        deal_amount = pd.Series()
        trade_price = pd.Series()
        trade_value = pd.Series()
        trade_cost = pd.Series()
        trade_dir = pd.Series()
        for _order_indicator in inner_order_indicators:
            inner_amount = inner_amount.add(_order_indicator["inner_amount"], fill_value=0)
            deal_amount = deal_amount.add(_order_indicator["deal_amount"], fill_value=0)
            trade_price = trade_price.add(
                _order_indicator["trade_price"] * _order_indicator["deal_amount"], fill_value=0
            )
            trade_value = trade_value.add(_order_indicator["trade_value"], fill_value=0)
            trade_cost = trade_cost.add(_order_indicator["trade_cost"], fill_value=0)
            trade_dir = trade_dir.add(_order_indicator["trade_dir"], fill_value=0)

        trade_dir = trade_dir.apply(Order.parse_dir)

        self.order_indicator["inner_amount"] = inner_amount
        self.order_indicator["deal_amount"] = deal_amount
        trade_price /= self.order_indicator["deal_amount"]
        self.order_indicator["trade_price"] = trade_price
        self.order_indicator["trade_value"] = trade_value
        self.order_indicator["trade_cost"] = trade_cost
        self.order_indicator["trade_dir"] = trade_dir

    def _update_trade_amount(self, outer_trade_decision: BaseTradeDecision):
        # NOTE: these indicator is designed for order execution, so the
        decision: List[Order] = outer_trade_decision.get_decision()
        if decision is None:
            self.order_indicator["amount"] = pd.Series()
        else:
            self.order_indicator["amount"] = pd.Series({order.stock_id: order.amount_delta for order in decision})

    def _agg_order_fulfill_rate(self):
        self.order_indicator["ffr"] = self.order_indicator["deal_amount"] / self.order_indicator["amount"]

    def _get_base_vol_pri(
        self,
        inst: str,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        direction: OrderDir,
        decision: BaseTradeDecision,
        trade_exchange: Exchange,
        pa_config: dict = {},
    ):
        """Get the base volume and price information"""

        agg = pa_config.get("agg", "twap").lower()
        price = pa_config.get("price", "deal_price").lower()

        if price == "deal_price":
            price_s = trade_exchange.get_deal_price(
                inst, trade_start_time, trade_end_time, direction=direction, method=None
            )
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # NOTE: there are some zeros in the trading price. These cases are known meaningless
        # for aligning the previous logic, remove it.
        # price_s = price_s.mask(np.isclose(price_s, 0))

        if agg == "vwap":
            volume_s = trade_exchange.get_volume(inst, trade_start_time, trade_end_time, method=None)
        elif agg == "twap":
            volume_s = pd.Series(1, index=price_s.index)
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # no sub executor on the lowest level
        # So range_limit an total step will all be None
        total_step = decision.total_step
        if total_step is None:
            total_step = 1
        range_limit = decision.get_range_limit(default_value=(0, total_step - 1))

        assert volume_s.shape[0] % total_step == 0, "The price series can't  be divided by step length"
        factor = volume_s.shape[0] // total_step

        slc = slice(range_limit[0] * factor, (range_limit[1] + 1) * factor)

        volume_s = volume_s.iloc[slc]
        price_s = price_s.iloc[slc]

        base_volume = volume_s.sum().item()
        base_price = ((price_s * volume_s).sum() / base_volume).item()

        return base_price, base_volume

    def _agg_base_price(
        self,
        inner_order_indicators: List[Dict[str, pd.Series]],
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]],
        trade_exchange: Exchange,
        pa_config: dict = {},
    ):
        """
        # NOTE:!!!!
        # Strong assumption!!!!!!
        # the correctness of the base_price relies on that the **same** exchange is used

        Parameters
        ----------
        inner_order_indicators : List[Dict[str, pd.Series]]
            the indicators of account of inner executor
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]],
            a list of decisions according to inner_order_indicators
        trade_exchange : Exchange
            for retrieving trading price
        pa_config : dict
            For example
            {
                "agg": "twap",  # "vwap"
                "price": "$close",  # TODO: this is not supported now!!!!!
                                    # default to use deal price of the exchange
            }

        """

        # TODO: I think there are potentials to be optimized
        trade_dir = self.order_indicator["trade_dir"]
        if len(trade_dir) > 0:
            bp_all, bv_all = [], []
            # <step, inst, (base_volume | base_price)>
            for oi, (dec, start, end) in zip(inner_order_indicators, decision_list):
                bp_s = oi.get("base_price", pd.Series()).reindex(trade_dir.index)
                bv_s = oi.get("base_volume", pd.Series()).reindex(trade_dir.index)
                bp_new, bv_new = {}, {}
                for pr, v, (inst, direction) in zip(bp_s.values, bv_s.values, trade_dir.items()):
                    if np.isnan(pr):
                        bp_new[inst], bv_new[inst] = self._get_base_vol_pri(
                            inst,
                            start,
                            end,
                            decision=dec,
                            direction=direction,
                            trade_exchange=trade_exchange,
                            pa_config=pa_config,
                        )
                    else:
                        bp_new[inst], bv_new[inst] = pr, v

                bp_new, bv_new = pd.Series(bp_new), pd.Series(bv_new)
                bp_all.append(bp_new)
                bv_all.append(bv_new)
            bp_all = pd.concat(bp_all, axis=1)
            bv_all = pd.concat(bv_all, axis=1)

            self.order_indicator["base_volume"] = bv_all.sum(axis=1)
            self.order_indicator["base_price"] = (bp_all * bv_all).sum(axis=1) / self.order_indicator["base_volume"]

    def _agg_order_price_advantage(self):
        if not self.order_indicator["trade_price"].empty:
            sign = 1 - self.order_indicator["trade_dir"] * 2
            self.order_indicator["pa"] = sign * (
                self.order_indicator["trade_price"] / self.order_indicator["base_price"] - 1
            )
        else:
            self.order_indicator["pa"] = pd.Series()

    def agg_order_indicators(
        self,
        inner_order_indicators: List[Dict[str, pd.Series]],
        decision_list: List[Tuple[BaseTradeDecision, pd.Timestamp, pd.Timestamp]],
        outer_trade_decision: BaseTradeDecision,
        trade_exchange: Exchange,
        indicator_config={},
    ):
        self._agg_order_trade_info(inner_order_indicators)
        self._update_trade_amount(outer_trade_decision)
        self._agg_order_fulfill_rate()
        pa_config = indicator_config.get("pa_config", {})
        self._agg_base_price(inner_order_indicators, decision_list, trade_exchange, pa_config=pa_config)
        self._agg_order_price_advantage()

    def _cal_trade_fulfill_rate(self, method="mean"):
        if method == "mean":
            return self.order_indicator["ffr"].mean()
        elif method == "amount_weighted":
            weights = self.order_indicator["deal_amount"].abs()
            return (self.order_indicator["ffr"] * weights).sum() / weights.sum()
        elif method == "value_weighted":
            weights = self.order_indicator["trade_value"].abs()
            return (self.order_indicator["ffr"] * weights).sum() / weights.sum()
        else:
            raise ValueError(f"method {method} is not supported!")

    def _cal_trade_price_advantage(self, method="mean"):
        pa_order = self.order_indicator["pa"]
        if isinstance(pa_order, (int, float)):
            # pa from atomic executor
            return pa_order

        if method == "mean":
            return pa_order.mean()
        elif method == "amount_weighted":
            weights = self.order_indicator["deal_amount"].abs()
            return (pa_order * weights).sum() / weights.sum()
        elif method == "value_weighted":
            weights = self.order_indicator["trade_value"].abs()
            return (pa_order * weights).sum() / weights.sum()
        else:
            raise ValueError(f"method {method} is not supported!")

    def _cal_trade_positive_rate(self):
        pa_order = self.order_indicator["pa"]
        if isinstance(pa_order, (int, float)):
            # pa from atomic executor
            return pa_order
        return (pa_order > 0).astype(int).sum() / pa_order.count()

    def _cal_trade_amount(self):
        return self.order_indicator["deal_amount"].abs().sum()

    def _cal_trade_value(self):
        return self.order_indicator["trade_value"].abs().sum()

    def _cal_trade_order_count(self):
        return self.order_indicator["amount"].count()

    def cal_trade_indicators(self, trade_start_time, freq, indicator_config={}):
        show_indicator = indicator_config.get("show_indicator", False)
        ffr_config = indicator_config.get("ffr_config", {})
        pa_config = indicator_config.get("pa_config", {})
        fulfill_rate = self._cal_trade_fulfill_rate(method=ffr_config.get("weight_method", "mean"))
        price_advantage = self._cal_trade_price_advantage(method=pa_config.get("weight_method", "mean"))
        positive_rate = self._cal_trade_positive_rate()
        trade_amount = self._cal_trade_amount()
        trade_value = self._cal_trade_value()
        order_count = self._cal_trade_order_count()
        self.trade_indicator["ffr"] = fulfill_rate
        self.trade_indicator["pa"] = price_advantage
        self.trade_indicator["pos"] = positive_rate
        self.trade_indicator["amount"] = trade_amount
        self.trade_indicator["value"] = trade_value
        self.trade_indicator["count"] = order_count
        if show_indicator:
            print(
                "[Indicator({}) {:%Y-%m-%d %H:%M:%S}]: FFR: {}, PA: {}, POS: {}".format(
                    freq, trade_start_time, fulfill_rate, price_advantage, positive_rate
                )
            )

    def get_order_indicator(self):
        return self.order_indicator

    def get_trade_indicator(self):
        return self.trade_indicator

    def generate_trade_indicators_dataframe(self):
        return pd.DataFrame.from_dict(self.trade_indicator_his, orient="index")
