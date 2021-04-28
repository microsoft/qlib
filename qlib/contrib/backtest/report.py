# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections import OrderedDict
import pandas as pd
import pathlib


class Report:
    # daily report of the account
    # contain those followings: returns, costs turnovers, accounts, cash, bench, value
    # update report
    def __init__(self):
        self.init_vars()

    def init_vars(self):
        self.accounts = OrderedDict()  # account postion value for each trade date
        self.returns = OrderedDict()  # daily return rate for each trade date
        self.turnovers = OrderedDict()  # turnover for each trade date
        self.costs = OrderedDict()  # trade cost for each trade date
        self.values = OrderedDict()  # value for each trade date
        self.cashes = OrderedDict()
        self.benches = OrderedDict()
        self.latest_report_time = None  # pd.TimeStamp

    def is_empty(self):
        return len(self.accounts) == 0

    def get_latest_date(self):
        return self.latest_report_time

    def get_latest_account_value(self):
        return self.accounts[self.latest_report_time]

    def update_report_record(
        self,
        trade_time=None,
        account_value=None,
        cash=None,
        return_rate=None,
        turnover_rate=None,
        cost_rate=None,
        stock_value=None,
        bench_value=None,
    ):
        # check data
        if None in [
            trade_time,
            account_value,
            cash,
            return_rate,
            turnover_rate,
            cost_rate,
            stock_value,
            bench_value
        ]:
            raise ValueError(
                "None in [trade_date, account_value, cash, return_rate, turnover_rate, cost_rate, stock_value, bench_value]"
            )
        # update report data
        self.accounts[trade_time] = account_value
        self.returns[trade_time] = return_rate
        self.turnovers[trade_time] = turnover_rate
        self.costs[trade_time] = cost_rate
        self.values[trade_time] = stock_value
        self.cashes[trade_time] = cash
        self.benches[trade_time] = bench_value
        # update latest_report_date
        self.latest_report_time = trade_time
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
                bench_value=r.loc[trade_time]["bench"]
            )
