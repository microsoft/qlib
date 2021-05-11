# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .account import Account


def backtest(start_time, end_time, trade_strategy, trade_env, benchmark, account):

    trade_account = Account(init_cash=account, benchmark=benchmark, start_time=start_time, end_time=end_time)
    trade_env.reset(start_time=start_time, end_time=end_time, trade_account=trade_account)
    trade_strategy.reset(start_time=start_time, end_time=end_time)

    _execute_state = trade_env.get_init_state()
    while not trade_env.finished():
        _order_list = trade_strategy.generate_order_list(_execute_state)
        _execute_state = trade_env.execute(_order_list)

    return trade_env.get_report()
