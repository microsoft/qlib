# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import numpy as np
import pandas as pd

from .account import Account

def backtest(trade_strategy, trade_env, benchmark, account):

    trade_account = Account(init_cash=account)
    trade_env.reset(start_time=start_time, end_time=end_time, trade_account=trade_account)
    trade_strategy.reset(start_time=start_time, end_time=end_time)

    trade_state = self.sub_env.get_init_state()
    while not trade_env.finished():
        _order_list = self.sub_strategy.generate_order(**trade_state)
        trade_state, trade_info = self.sub_env.execute(sub_order_list)
    
    report_df = trade_account.report.generate_report_dataframe()
    positions = trade_account.get_positions()
    report_dict = {"report_df": report_df, "positions": positions}

    return report_dict

