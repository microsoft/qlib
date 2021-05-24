# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def backtest(start_time, end_time, trade_strategy, trade_executor):

    trade_executor.reset(start_time=start_time, end_time=end_time)
    level_infra = trade_executor.get_level_infra()
    trade_strategy.reset(level_infra=level_infra)

    sub_execute_state = trade_executor.get_init_state()
    while not trade_executor.finished():
        sub_trade_decision = trade_strategy.generate_trade_decision(sub_execute_state)
        sub_execute_state = trade_executor.execute(sub_trade_decision)

    return trade_executor.get_report()


def collect_data(start_time, end_time, trade_strategy, trade_executor):

    trade_executor.reset(start_time=start_time, end_time=end_time)
    level_infra = trade_executor.get_level_infra()
    trade_strategy.reset(level_infra=level_infra)

    sub_execute_state = trade_executor.get_init_state()
    while not trade_executor.finished():
        sub_trade_decision = trade_strategy.generate_trade_decision(sub_execute_state)
        sub_execute_state = yield from trade_executor.collect_data(sub_trade_decision)

    return trade_executor.get_report()
