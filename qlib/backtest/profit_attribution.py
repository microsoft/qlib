# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
from .position import Position
from ...data import D
from ...config import C
import datetime
from pathlib import Path


def get_benchmark_weight(
    bench,
    start_date=None,
    end_date=None,
    path=None,
):
    """get_benchmark_weight

    get the stock weight distribution of the benchmark

    :param bench:
    :param start_date:
    :param end_date:
    :param path:

    :return: The weight distribution of the the benchmark described by a pandas dataframe
             Every row corresponds to a trading day.
             Every column corresponds to a stock.
             Every cell represents the strategy.

    """
    if not path:
        path = Path(C.get_data_path()).expanduser() / "raw" / "AIndexMembers" / "weights.csv"
    # TODO: the storage of weights should be implemented in a more elegent way
    # TODO: The benchmark is not consistant with the filename in instruments.
    bench_weight_df = pd.read_csv(path, usecols=["code", "date", "index", "weight"])
    bench_weight_df = bench_weight_df[bench_weight_df["index"] == bench]
    bench_weight_df["date"] = pd.to_datetime(bench_weight_df["date"])
    if start_date is not None:
        bench_weight_df = bench_weight_df[bench_weight_df.date >= start_date]
    if end_date is not None:
        bench_weight_df = bench_weight_df[bench_weight_df.date <= end_date]
    bench_stock_weight = bench_weight_df.pivot_table(index="date", columns="code", values="weight") / 100.0
    return bench_stock_weight


def get_stock_weight_df(positions):
    """get_stock_weight_df
    :param positions: Given a positions from backtest result.
    :return:          A weight distribution for the position
    """
    stock_weight = []
    index = []
    for date in sorted(positions.keys()):
        pos = positions[date]
        if isinstance(pos, dict):
            pos = Position(position_dict=pos)
        index.append(date)
        stock_weight.append(pos.get_stock_weight_dict(only_stock=True))
    return pd.DataFrame(stock_weight, index=index)


def decompose_portofolio_weight(stock_weight_df, stock_group_df):
    """decompose_portofolio_weight

    '''
    :param stock_weight_df: a pandas dataframe to describe the portofolio by weight.
                    every row corresponds to a  day
                    every column corresponds to a stock.
                    Here is an example below.
                    code        SH600004  SH600006  SH600017  SH600022  SH600026  SH600037  \
                    date
                    2016-01-05  0.001543  0.001570  0.002732  0.001320  0.003000       NaN
                    2016-01-06  0.001538  0.001569  0.002770  0.001417  0.002945       NaN
                    ....
    :param stock_group_df: a pandas dataframe to describe  the stock group.
                    every row corresponds to a  day
                    every column corresponds to a stock.
                    the value in the cell repreponds the group id.
                    Here is a example by for stock_group_df for industry. The value is the industry code
                    instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008  \
                    datetime
                    2016-01-05  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
                    2016-01-06  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
                    ...
    :return:        Two dict will be returned.  The group_weight and the stock_weight_in_group.
                    The key is the group. The value is a Series or Dataframe to describe the weight of group or weight of stock
    """
    all_group = np.unique(stock_group_df.values.flatten())
    all_group = all_group[~np.isnan(all_group)]

    group_weight = {}
    stock_weight_in_group = {}
    for group_key in all_group:
        group_mask = stock_group_df == group_key
        group_weight[group_key] = stock_weight_df[group_mask].sum(axis=1)
        stock_weight_in_group[group_key] = stock_weight_df[group_mask].divide(group_weight[group_key], axis=0)
    return group_weight, stock_weight_in_group


def decompose_portofolio(stock_weight_df, stock_group_df, stock_ret_df):
    """
    :param stock_weight_df: a pandas dataframe to describe the portofolio by weight.
                    every row corresponds to a  day
                    every column corresponds to a stock.
                    Here is an example below.
                    code        SH600004  SH600006  SH600017  SH600022  SH600026  SH600037  \
                    date
                    2016-01-05  0.001543  0.001570  0.002732  0.001320  0.003000       NaN
                    2016-01-06  0.001538  0.001569  0.002770  0.001417  0.002945       NaN
                    2016-01-07  0.001555  0.001546  0.002772  0.001393  0.002904       NaN
                    2016-01-08  0.001564  0.001527  0.002791  0.001506  0.002948       NaN
                    2016-01-11  0.001597  0.001476  0.002738  0.001493  0.003043       NaN
                    ....

    :param stock_group_df: a pandas dataframe to describe  the stock group.
                    every row corresponds to a  day
                    every column corresponds to a stock.
                    the value in the cell repreponds the group id.
                    Here is a example by for stock_group_df for industry. The value is the industry code
                    instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008  \
                    datetime
                    2016-01-05  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
                    2016-01-06  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
                    2016-01-07  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
                    2016-01-08  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
                    2016-01-11  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0
                    ...

    :param stock_ret_df:   a pandas dataframe to describe the stock return.
                    every row corresponds to a day
                    every column corresponds to a stock.
                    the value in the cell repreponds the return of the group.
                    Here is a example by for stock_ret_df.
                    instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008  \
                    datetime
                    2016-01-05  0.007795  0.022070  0.099099  0.024707  0.009473  0.016216
                    2016-01-06 -0.032597 -0.075205 -0.098361 -0.098985 -0.099707 -0.098936
                    2016-01-07 -0.001142  0.022544  0.100000  0.004225  0.000651  0.047226
                    2016-01-08 -0.025157 -0.047244 -0.038567 -0.098177 -0.099609 -0.074408
                    2016-01-11  0.023460  0.004959 -0.034384  0.018663  0.014461  0.010962
                    ...

    :return: It will decompose the portofolio to the group weight and group return.
    """
    all_group = np.unique(stock_group_df.values.flatten())
    all_group = all_group[~np.isnan(all_group)]

    group_weight, stock_weight_in_group = decompose_portofolio_weight(stock_weight_df, stock_group_df)

    group_ret = {}
    for group_key in stock_weight_in_group:
        stock_weight_in_group_start_date = min(stock_weight_in_group[group_key].index)
        stock_weight_in_group_end_date = max(stock_weight_in_group[group_key].index)

        temp_stock_ret_df = stock_ret_df[
            (stock_ret_df.index >= stock_weight_in_group_start_date)
            & (stock_ret_df.index <= stock_weight_in_group_end_date)
        ]

        group_ret[group_key] = (temp_stock_ret_df * stock_weight_in_group[group_key]).sum(axis=1)
        # If no weight is assigned, then the return of group will be np.nan
        group_ret[group_key][group_weight[group_key] == 0.0] = np.nan

    group_weight_df = pd.DataFrame(group_weight)
    group_ret_df = pd.DataFrame(group_ret)
    return group_weight_df, group_ret_df


def get_daily_bin_group(bench_values, stock_values, group_n):
    """get_daily_bin_group
    Group the values of the stocks of benchmark into several bins in a day.
    Put the stocks into these bins.

    :param bench_values: A series contains the value of stocks in benchmark.
                         The index is the stock code.
    :param stock_values: A series contains the value of stocks of your portofolio
                         The index is the stock code.
    :param group_n:      Bins will be produced

    :return:             A series with the same size and index as the stock_value.
                         The value in the series is the group id of the bins.
                         The No.1 bin contains the biggest values.
    """
    stock_group = stock_values.copy()

    # get the bin split points based on the daily proportion of benchmark
    split_points = np.percentile(bench_values[~bench_values.isna()], np.linspace(0, 100, group_n + 1))
    # Modify the biggest uppper bound and smallest lowerbound
    split_points[0], split_points[-1] = -np.inf, np.inf
    for i, (lb, up) in enumerate(zip(split_points, split_points[1:])):
        stock_group.loc[stock_values[(stock_values >= lb) & (stock_values < up)].index] = group_n - i
    return stock_group


def get_stock_group(stock_group_field_df, bench_stock_weight_df, group_method, group_n=None):
    if group_method == "category":
        # use the value of the benchmark as the category
        return stock_group_field_df
    elif group_method == "bins":
        assert group_n is not None
        # place the values into `group_n` fields.
        # Each bin corresponds to a category.
        new_stock_group_df = stock_group_field_df.copy().loc[
            bench_stock_weight_df.index.min() : bench_stock_weight_df.index.max()
        ]
        for idx, row in (~bench_stock_weight_df.isna()).iterrows():
            bench_values = stock_group_field_df.loc[idx, row[row].index]
            new_stock_group_df.loc[idx] = get_daily_bin_group(
                bench_values, stock_group_field_df.loc[idx], group_n=group_n
            )
        return new_stock_group_df


def brinson_pa(
    positions,
    bench="SH000905",
    group_field="industry",
    group_method="category",
    group_n=None,
    deal_price="vwap",
):
    """brinson profit attribution

    :param positions: The position produced by the backtest class
    :param bench: The benchmark for comparing. TODO: if no benchmark is set, the equal-weighted is used.
    :param group_field: The field used to set the group for assets allocation.
                        `industry` and `market_value` is often used.
    :param group_method: 'category' or 'bins'. The method used to set the group for asstes allocation
                         `bin` will split the value into `group_n` bins and each bins represents a group
    :param group_n: . Only used when group_method == 'bins'.

    :return:
        A dataframe with three columns: RAA(excess Return of Assets Allocation),  RSS(excess Return of Stock Selectino),  RTotal(Total excess Return)
                                        Every row corresponds to a trading day, the value corresponds to the next return for this trading day
        The middle info of brinson profit attribution
    """
    # group_method will decide how to group the group_field.
    dates = sorted(positions.keys())

    start_date, end_date = min(dates), max(dates)

    bench_stock_weight = get_benchmark_weight(bench, start_date, end_date)

    # The attributes for allocation will not
    if not group_field.startswith("$"):
        group_field = "$" + group_field
    if not deal_price.startswith("$"):
        deal_price = "$" + deal_price

    # FIXME: In current version.  Some attributes(such as market_value) of some
    # suspend stock is NAN. So we have to get more date to forward fill the NAN
    shift_start_date = start_date - datetime.timedelta(days=250)
    instruments = D.list_instruments(
        D.instruments(market="all"),
        start_time=shift_start_date,
        end_time=end_date,
        as_list=True,
    )
    stock_df = D.features(
        instruments,
        [group_field, deal_price],
        start_time=shift_start_date,
        end_time=end_date,
        freq="day",
    )
    stock_df.columns = [group_field, "deal_price"]

    stock_group_field = stock_df[group_field].unstack().T
    # FIXME: some attributes of some suspend stock is NAN.
    stock_group_field = stock_group_field.fillna(method="ffill")
    stock_group_field = stock_group_field.loc[start_date:end_date]

    stock_group = get_stock_group(stock_group_field, bench_stock_weight, group_method, group_n)

    deal_price_df = stock_df["deal_price"].unstack().T
    deal_price_df = deal_price_df.fillna(method="ffill")

    # NOTE:
    # The return will be slightly different from the of the return in the report.
    # Here the position are adjusted at the end of the trading day with close
    stock_ret = (deal_price_df - deal_price_df.shift(1)) / deal_price_df.shift(1)
    stock_ret = stock_ret.shift(-1).loc[start_date:end_date]

    port_stock_weight_df = get_stock_weight_df(positions)

    # decomposing the portofolio
    port_group_weight_df, port_group_ret_df = decompose_portofolio(port_stock_weight_df, stock_group, stock_ret)
    bench_group_weight_df, bench_group_ret_df = decompose_portofolio(bench_stock_weight, stock_group, stock_ret)

    # if the group return of the portofolio is NaN, replace it with the market
    # value
    mod_port_group_ret_df = port_group_ret_df.copy()
    mod_port_group_ret_df[mod_port_group_ret_df.isna()] = bench_group_ret_df

    Q1 = (bench_group_weight_df * bench_group_ret_df).sum(axis=1)
    Q2 = (port_group_weight_df * bench_group_ret_df).sum(axis=1)
    Q3 = (bench_group_weight_df * mod_port_group_ret_df).sum(axis=1)
    Q4 = (port_group_weight_df * mod_port_group_ret_df).sum(axis=1)

    return (
        pd.DataFrame(
            {
                "RAA": Q2 - Q1,  # The excess profit from the assets allocation
                "RSS": Q3 - Q1,  # The excess profit from the stocks selection
                # The excess profit from the interaction of assets allocation and stocks selection
                "RIN": Q4 - Q3 - Q2 + Q1,
                "RTotal": Q4 - Q1,  # The totoal excess profit
            }
        ),
        {
            "port_group_ret": port_group_ret_df,
            "port_group_weight": port_group_weight_df,
            "bench_group_ret": bench_group_ret_df,
            "bench_group_weight": bench_group_weight_df,
            "stock_group": stock_group,
            "bench_stock_weight": bench_stock_weight,
            "port_stock_weight": port_stock_weight_df,
            "stock_ret": stock_ret,
        },
    )
