# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections import OrderedDict
from logging import warning
import pandas as pd
import pathlib
import warnings

from pandas.core.frame import DataFrame

from ..utils.resam import parse_freq, resam_ts_data
from ..data import D


class Report:
    # daily report of the account
    # contain those followings: returns, costs turnovers, accounts, cash, bench, value
    # update report
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
        self.accounts = OrderedDict()  # account postion value for each trade date
        self.returns = OrderedDict()  # daily return rate for each trade date
        self.turnovers = OrderedDict()  # turnover for each trade date
        self.costs = OrderedDict()  # trade cost for each trade date
        self.values = OrderedDict()  # value for each trade date
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
        benchmark = benchmark_config.get("benchmark", "SH000300")
        if isinstance(benchmark, pd.Series):
            return benchmark
        else:
            start_time = benchmark_config.get("start_time", None)
            end_time = benchmark_config.get("end_time", None)

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
                        _temp_result = D.features(_codes, fields, start_time, end_time, freq="1min", disk_cache=1)
                elif norm_freq == "minute":
                    _temp_result = D.features(_codes, fields, start_time, end_time, freq="1min", disk_cache=1)
                else:
                    raise ValueError(f"benchmark freq {freq} is not supported")
            if len(_temp_result) == 0:
                raise ValueError(f"The benchmark {_codes} does not exist. Please provide the right benchmark")
            return _temp_result.groupby(level="datetime")[_temp_result.columns.tolist()[0]].mean().fillna(0)

    def _sample_benchmark(self, bench, trade_start_time, trade_end_time):
        def cal_change(x):
            return (x + 1).prod() - 1

        _ret = resam_ts_data(bench, trade_start_time, trade_end_time, method=cal_change)
        return 0.0 if _ret is None else _ret

    def is_empty(self):
        return len(self.accounts) == 0

    def get_latest_date(self):
        return self.latest_report_time

    def get_latest_account_value(self):
        return self.accounts[self.latest_report_time]

    def update_report_record(
        self,
        trade_start_time=None,
        trade_end_time=None,
        account_value=None,
        cash=None,
        return_rate=None,
        turnover_rate=None,
        cost_rate=None,
        stock_value=None,
    ):
        # check data
        if None in [
            trade_start_time,
            trade_end_time,
            account_value,
            cash,
            return_rate,
            turnover_rate,
            cost_rate,
            stock_value,
        ]:
            raise ValueError(
                "None in [trade_start_time, trade_end_time, account_value, cash, return_rate, turnover_rate, cost_rate, stock_value]"
            )
        # update report data
        self.accounts[trade_start_time] = account_value
        self.returns[trade_start_time] = return_rate
        self.turnovers[trade_start_time] = turnover_rate
        self.costs[trade_start_time] = cost_rate
        self.values[trade_start_time] = stock_value
        self.cashes[trade_start_time] = cash
        self.benches[trade_start_time] = self._sample_benchmark(self.bench, trade_start_time, trade_end_time)
        # update latest_report_date
        self.latest_report_time = trade_start_time
        # finish daily report update

    def generate_report_dataframe(self):
        report = pd.DataFrame()
        report["account"] = pd.Series(self.accounts)
        report["return"] = pd.Series(self.returns)
        report["turnover"] = pd.Series(self.turnovers)
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
        columns = ['account', 'return', 'turnover', 'cost', 'value', 'cash', 'bench']
            :param
                path: str/ pathlib.Path()
        """
        path = pathlib.Path(path)
        r = pd.read_csv(open(path, "rb"), index_col=0)
        r.index = pd.DatetimeIndex(r.index)

        index = r.index
        self.init_vars()
        for trade_time in index:
            self.update_report_record(
                trade_time=trade_time,
                account_value=r.loc[trade_time]["account"],
                cash=r.loc[trade_time]["cash"],
                return_rate=r.loc[trade_time]["return"],
                turnover_rate=r.loc[trade_time]["turnover"],
                cost_rate=r.loc[trade_time]["cost"],
                stock_value=r.loc[trade_time]["value"],
                bench_value=r.loc[trade_time]["bench"],
            )
