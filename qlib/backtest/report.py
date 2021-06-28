# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections import OrderedDict
from logging import warning
import pandas as pd
import pathlib
import warnings
from pandas.core import groupby

from pandas.core.frame import DataFrame

from ..utils.time import Freq
from ..utils.resam import resam_ts_data, get_higher_eq_freq_feature
from ..data import D
from ..tests.config import CSI300_BENCH


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
            _codes = benchmark if isinstance(benchmark, list) else [benchmark]
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
    def __init__(self):
        self.order_indicator_his = OrderedDict()
        self.order_indicator = OrderedDict()
        self.trade_indicator_his = OrderedDict()
        self.trade_indicator = OrderedDict()

    def clear(self):
        self.order_indicator = OrderedDict()
        self.trade_indicator = OrderedDict()

    def record(self, trade_start_time):
        self.order_indicator_his[trade_start_time] = self.order_indicator
        self.trade_indicator_his[trade_start_time] = self.trade_indicator

    def _update_order_trade_info(self, trade_info: list):
        amount = dict()
        deal_amount = dict()
        trade_price = dict()
        trade_value = dict()
        trade_cost = dict()

        for order, _trade_val, _trade_cost, _trade_price in trade_info:
            amount[order.stock_id] = order.amount * (order.direction * 2 - 1)
            deal_amount[order.stock_id] = order.deal_amount * (order.direction * 2 - 1)
            trade_price[order.stock_id] = _trade_price
            trade_value[order.stock_id] = _trade_val * (order.direction * 2 - 1)
            trade_cost[order.stock_id] = _trade_cost

        self.order_indicator["amount"] = pd.Series(amount)
        self.order_indicator["deal_amount"] = pd.Series(deal_amount)
        self.order_indicator["trade_price"] = pd.Series(trade_price)
        self.order_indicator["trade_value"] = pd.Series(trade_value)
        self.order_indicator["trade_cost"] = pd.Series(trade_cost)

    def _update_order_fulfill_rate(self):
        self.order_indicator["ffr"] = self.order_indicator["deal_amount"] / self.order_indicator["amount"]

    def _update_order_price_advantage(self, trade_exchange, trade_start_time, trade_end_time):
        self.order_indicator["base_price"] = self.order_indicator["trade_price"]
        instruments = list(self.order_indicator["base_price"].index)
        self.order_indicator["volume"] = pd.Series(
            [
                trade_exchange.get_volume(stock_id=inst, start_time=trade_start_time, end_time=trade_end_time)
                for inst in instruments
            ],
            index=instruments,
        )
        self.order_indicator["pa"] = (
            self.order_indicator["trade_price"] - self.order_indicator["base_price"]
        ) / self.order_indicator["base_price"]

    def _agg_order_trade_info(self, inner_order_indicators):
        amount = pd.Series()
        deal_amount = pd.Series()
        trade_price = pd.Series()
        trade_value = pd.Series()
        trade_cost = pd.Series()
        for _order_indicator in inner_order_indicators:
            amount = amount.add(_order_indicator["amount"], fill_value=0)
            deal_amount = deal_amount.add(_order_indicator["deal_amount"], fill_value=0)
            trade_price = trade_price.add(
                _order_indicator["trade_price"] * _order_indicator["deal_amount"], fill_value=0
            )
            trade_value = trade_value.add(_order_indicator["trade_value"], fill_value=0)
            trade_cost = trade_cost.add(_order_indicator["trade_cost"], fill_value=0)

        self.order_indicator["amount"] = amount
        self.order_indicator["deal_amount"] = deal_amount
        trade_price /= self.order_indicator["deal_amount"]
        self.order_indicator["trade_price"] = trade_price
        self.order_indicator["trade_value"] = trade_value
        self.order_indicator["trade_cost"] = trade_cost

    def _agg_order_fulfill_rate(self):
        self.order_indicator["ffr"] = self.order_indicator["deal_amount"] / self.order_indicator["amount"]

    def _agg_order_price_advantage(self, inner_order_indicators, base_price="twap"):
        base_price = base_price.lower()
        volume = pd.Series()
        for _order_indicator in inner_order_indicators:
            volume = volume.add(_order_indicator["volume"], fill_value=0)
        self.order_indicator["volume"] = volume

        if base_price == "twap":
            base_price = pd.Series()
            price_count = pd.Series()
            for _order_indicator in inner_order_indicators:
                base_price = base_price.add(_order_indicator["base_price"], fill_value=0)
                price_count = price_count.add(pd.Series(1, index=_order_indicator["base_price"].index), fill_value=0)
            base_price /= price_count
            self.order_indicator["base_price"] = base_price

        elif base_price == "vwap":
            base_price = pd.Series()
            for _order_indicator in inner_order_indicators:
                base_price = base_price.add(_order_indicator["base_price"] * _order_indicator["volume"], fill_value=0)
            base_price /= self.order_indicator["volume"]
            self.order_indicator["base_price"] = base_price

        else:
            raise ValueError(f"base_price {base_price} is not supported!")

        self.order_indicator["pa"] = self.order_indicator["trade_price"] / self.order_indicator["base_price"] - 1
        # print("trade_price", self.order_indicator["trade_price"], "base_price", self.order_indicator["base_price"], "pa", self.order_indicator["pa"]* (2 * (self.order_indicator["amount"] < 0).astype(int) - 1))

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
        pa_order = self.order_indicator["pa"] * (2 * (self.order_indicator["amount"] < 0).astype(int) - 1)
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
        pa_order = self.order_indicator["pa"] * (2 * (self.order_indicator["amount"] < 0).astype(int) - 1)
        return (pa_order > 0).astype(int).sum() / pa_order.count()

    def _cal_trade_amount(self):
        return self.order_indicator["deal_amount"].abs().sum()

    def _cal_trade_value(self):
        return self.order_indicator["trade_value"].abs().sum()

    def _cal_trade_order_count(self):
        return self.order_indicator["amount"].count()

    def update_order_indicators(self, trade_start_time, trade_end_time, trade_info, trade_exchange):
        self._update_order_trade_info(trade_info=trade_info)
        self._update_order_fulfill_rate()
        self._update_order_price_advantage(trade_exchange, trade_start_time, trade_end_time)

    def agg_order_indicators(self, inner_order_indicators, indicator_config={}):
        self._agg_order_trade_info(inner_order_indicators)
        self._agg_order_fulfill_rate()
        pa_config = indicator_config.get("pa_config", {})
        self._agg_order_price_advantage(inner_order_indicators, base_price=pa_config.get("base_price", "twap"))

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
