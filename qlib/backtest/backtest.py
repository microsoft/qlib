# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def backtest(start_time, end_time, trade_strategy, trade_executor):

    trade_executor.reset(start_time=start_time, end_time=end_time)
    level_infra = trade_executor.get_level_infra()
    trade_strategy.reset(level_infra=level_infra)

    _execute_result = None
    while not trade_executor.finished():
        _trade_decision = trade_strategy.generate_trade_decision(_execute_result)
        _execute_result = trade_executor.execute(_trade_decision)

    return trade_executor.get_report()


def collect_data(start_time, end_time, trade_strategy, trade_executor):

    trade_executor.reset(start_time=start_time, end_time=end_time)
    level_infra = trade_executor.get_level_infra()
    trade_strategy.reset(level_infra=level_infra)

    _execute_result = None
    while not trade_executor.finished():
        _trade_decision = trade_strategy.generate_trade_decision(_execute_result)
        _execute_result = yield from trade_executor.collect_data(_trade_decision)

    return trade_executor.get_report()
