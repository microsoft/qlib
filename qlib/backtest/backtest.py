# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def backtest_loop(start_time, end_time, trade_strategy, trade_executor):
    """backtest funciton for the interaction of the outermost strategy and executor in the nested decison execution

    Parameters
    ----------
    start_time : pd.Timestamp|str
        closed start time for backtest
    end_time : pd.Timestamp|str
        closed end time for backtest
    trade_strategy : BaseStrategy
        the outermost portfolio strategy
    trade_executor : BaseExecutor
        the outermost executor

    Returns
    -------
    report: Report
        it records the trading report information
    """
    trade_executor.reset(start_time=start_time, end_time=end_time)
    level_infra = trade_executor.get_level_infra()
    trade_strategy.reset(level_infra=level_infra)

    _execute_result = None
    while not trade_executor.finished():
        _trade_decision = trade_strategy.generate_trade_decision(_execute_result)
        _execute_result = trade_executor.execute(_trade_decision)

    return trade_executor.get_report()


def collect_data_loop(start_time, end_time, trade_strategy, trade_executor):
    """Generator for collecting the trade decision data for rl training

    Yields
    -------
    object
        trade decision
    """
    trade_executor.reset(start_time=start_time, end_time=end_time)
    level_infra = trade_executor.get_level_infra()
    trade_strategy.reset(level_infra=level_infra)

    _execute_result = None
    while not trade_executor.finished():
        _trade_decision = trade_strategy.generate_trade_decision(_execute_result)
        _execute_result = yield from trade_executor.collect_data(_trade_decision)
