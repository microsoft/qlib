# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def backtest(start_time, end_time, trade_strategy, trade_env):

    trade_env.reset(start_time=start_time, end_time=end_time)
    trade_strategy.reset(start_time=start_time, end_time=end_time)

    _execute_state = trade_env.get_init_state()
    while not trade_env.finished():
        _order_list = trade_strategy.generate_order_list(_execute_state)
        _execute_state = trade_env.execute(_order_list)

    return trade_env.get_report()
