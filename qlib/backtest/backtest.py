# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from ..utils.resam import parse_freq


def backtest_loop(start_time, end_time, trade_strategy, trade_executor):
    """backtest funciton for the interaction of the outermost strategy and executor in the nested decison execution

    Returns
    -------
    report: Report
        it records the trading report information
    """
    return_value = {}
    for _decison in collect_data_loop(start_time, end_time, trade_strategy, trade_executor, return_value):
        pass
    return return_value.get("report"), return_value.get("indicator")


def collect_data_loop(start_time, end_time, trade_strategy, trade_executor, return_value: dict = None):
    """Generator for collecting the trade decision data for rl training

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
    return_value : dict
        used for backtest_loop

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

    if return_value is not None:
        all_executors = trade_executor.get_all_executors()

        all_reports = {
            "{}{}".format(*parse_freq(_executor.time_per_step)): _executor.get_report()
            for _executor in all_executors
            if _executor.generate_report
        }
        all_indicators = {
            "{}{}".format(
                *parse_freq(_executor.time_per_step)
            ): _executor.get_trade_indicator().generate_trade_indicators_dataframe()
            for _executor in all_executors
        }
        return_value.update({"report": all_reports, "indicator": all_indicators})
