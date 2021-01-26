# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
from ...utils import get_date_by_shift, get_date_range
from ...data import D
from .account import Account
from ...config import C
from ...log import get_module_logger
from ...data.dataset.utils import get_level_index

LOG = get_module_logger("backtest")


def backtest(pred, strategy, executor, trade_exchange, shift, verbose, account, benchmark, return_order):
    """Parameters
    ----------
    pred : pandas.DataFrame
        predict should has <datetime, instrument> index and one `score` column
        Qlib want to support multi-singal strategy in the future. So pd.Series is not used.
    strategy : Strategy()
        strategy part for backtest
    trade_exchange : Exchange()
        exchage for backtest
    shift : int
        whether to shift prediction by one day
    verbose : bool
        whether to print log
    account : float
        init account value
    benchmark : str/list/pd.Series
            `benchmark` is pd.Series, `index` is trading date; the value T is the change from T-1 to T.
                example:
                    print(D.features(D.instruments('csi500'), ['$close/Ref($close, 1)-1'])['$close/Ref($close, 1)-1'].head())
                        2017-01-04    0.011693
                        2017-01-05    0.000721
                        2017-01-06   -0.004322
                        2017-01-09    0.006874
                        2017-01-10   -0.003350

            `benchmark` is list, will use the daily average change of the stock pool in the list as the 'bench'.
            `benchmark` is str, will use the daily change as the 'bench'.
        benchmark code, default is SH000905 CSI500
    """
    # Convert format if the input format is not expected
    if get_level_index(pred, level="datetime") == 1:
        pred = pred.swaplevel().sort_index()
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")

    trade_account = Account(init_cash=account)
    _pred_dates = pred.index.get_level_values(level="datetime")
    predict_dates = D.calendar(start_time=_pred_dates.min(), end_time=_pred_dates.max())
    if isinstance(benchmark, pd.Series):
        bench = benchmark
    else:
        _codes = benchmark if isinstance(benchmark, list) else [benchmark]
        _temp_result = D.features(
            _codes,
            ["$close/Ref($close,1)-1"],
            predict_dates[0],
            get_date_by_shift(predict_dates[-1], shift=shift),
            disk_cache=1,
        )
        if len(_temp_result) == 0:
            raise ValueError(f"The benchmark {_codes} does not exist. Please provide the right benchmark")
        bench = _temp_result.groupby(level="datetime")[_temp_result.columns.tolist()[0]].mean()

    trade_dates = np.append(predict_dates[shift:], get_date_range(predict_dates[-1], left_shift=1, right_shift=shift))
    if return_order:
        multi_order_list = []
    # trading apart
    for pred_date, trade_date in zip(predict_dates, trade_dates):
        # for loop predict date and trading date
        # print
        if verbose:
            LOG.info("[I {:%Y-%m-%d}]: trade begin.".format(trade_date))

        # 1. Load the score_series at pred_date
        try:
            score = pred.loc(axis=0)[pred_date, :]  # (trade_date, stock_id) multi_index, score in pdate
            score_series = score.reset_index(level="datetime", drop=True)[
                "score"
            ]  # pd.Series(index:stock_id, data: score)
        except KeyError:
            LOG.warning("No score found on predict date[{:%Y-%m-%d}]".format(trade_date))
            score_series = None

        if score_series is not None and score_series.count() > 0:  # in case of the scores are all None
            # 2. Update your strategy (and model)
            strategy.update(score_series, pred_date, trade_date)

            # 3. Generate order list
            order_list = strategy.generate_order_list(
                score_series=score_series,
                current=trade_account.current,
                trade_exchange=trade_exchange,
                pred_date=pred_date,
                trade_date=trade_date,
            )
        else:
            order_list = []
        if return_order:
            multi_order_list.append((trade_account, order_list, trade_date))
        # 4. Get result after executing order list
        # NOTE: The following operation will modify order.amount.
        # NOTE: If it is buy and the cash is insufficient, the tradable amount will be recalculated
        trade_info = executor.execute(trade_account, order_list, trade_date)

        # 5. Update account information according to transaction
        update_account(trade_account, trade_info, trade_exchange, trade_date)

    # generate backtest report
    report_df = trade_account.report.generate_report_dataframe()
    report_df["bench"] = bench
    positions = trade_account.get_positions()

    report_dict = {"report_df": report_df, "positions": positions}
    if return_order:
        report_dict.update({"order_list": multi_order_list})
    return report_dict


def update_account(trade_account, trade_info, trade_exchange, trade_date):
    """Update the account and strategy
    Parameters
    ----------
    trade_account : Account()
    trade_info : list of [Order(), float, float, float]
        (order, trade_val, trade_cost, trade_price), trade_info with out factor
    trade_exchange : Exchange()
        used to get the $close_price at trade_date to update account
    trade_date : pd.Timestamp
    """
    # update account
    for [order, trade_val, trade_cost, trade_price] in trade_info:
        if order.deal_amount == 0:
            continue
        trade_account.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)
    # at the end of trade date, update the account based the $close_price of stocks.
    trade_account.update_daily_end(today=trade_date, trader=trade_exchange)
