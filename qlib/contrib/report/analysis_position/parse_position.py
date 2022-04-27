# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd


from ....backtest.profit_attribution import get_stock_weight_df


def parse_position(position: dict = None) -> pd.DataFrame:
    """Parse position dict to position DataFrame

    :param position: position data
    :return: position DataFrame;


        .. code-block:: python

            position_df = parse_position(positions)
            print(position_df.head())
            # status: 0-hold, -1-sell, 1-buy

                                        amount      cash      count    price status weight
            instrument  datetime
            SZ000547    2017-01-04  44.154290   211405.285654   1   205.189575  1   0.031255
            SZ300202    2017-01-04  60.638845   211405.285654   1   154.356506  1   0.032290
            SH600158    2017-01-04  46.531681   211405.285654   1   153.895142  1   0.024704
            SH600545    2017-01-04  197.173093  211405.285654   1   48.607037   1   0.033063
            SZ000930    2017-01-04  103.938300  211405.285654   1   80.759453   1   0.028958


    """

    position_weight_df = get_stock_weight_df(position)
    # If the day does not exist, use the last weight
    position_weight_df.fillna(method="ffill", inplace=True)

    previous_data = {"date": None, "code_list": []}

    result_df = pd.DataFrame()
    for _trading_date, _value in position.items():
        # pd_date type: pd.Timestamp
        _cash = _value.pop("cash")
        for _item in ["now_account_value"]:
            if _item in _value:
                _value.pop(_item)

        _trading_day_df = pd.DataFrame.from_dict(_value, orient="index")
        _trading_day_df["weight"] = position_weight_df.loc[_trading_date]
        _trading_day_df["cash"] = _cash
        _trading_day_df["date"] = _trading_date
        # status: 0-hold, -1-sell, 1-buy
        _trading_day_df["status"] = 0

        # T not exist, T-1 exist, T sell
        _cur_day_sell = set(previous_data["code_list"]) - set(_trading_day_df.index)
        # T exist, T-1 not exist, T buy
        _cur_day_buy = set(_trading_day_df.index) - set(previous_data["code_list"])

        # Trading day buy
        _trading_day_df.loc[_trading_day_df.index.isin(_cur_day_buy), "status"] = 1

        # Trading day sell
        if not result_df.empty:
            _trading_day_sell_df = result_df.loc[
                (result_df["date"] == previous_data["date"]) & (result_df.index.isin(_cur_day_sell))
            ].copy()
            if not _trading_day_sell_df.empty:
                _trading_day_sell_df["status"] = -1
                _trading_day_sell_df["date"] = _trading_date
                _trading_day_df = pd.concat([_trading_day_df, _trading_day_sell_df], sort=False)

        result_df = pd.concat([result_df, _trading_day_df], sort=True)

        previous_data = dict(
            date=_trading_date,
            code_list=_trading_day_df[_trading_day_df["status"] != -1].index,
        )

    result_df.reset_index(inplace=True)
    result_df.rename(columns={"date": "datetime", "index": "instrument"}, inplace=True)
    return result_df.set_index(["instrument", "datetime"])


def _add_label_to_position(position_df: pd.DataFrame, label_data: pd.DataFrame) -> pd.DataFrame:
    """Concat position with custom label

    :param position_df: position DataFrame
    :param label_data:
    :return: concat result
    """

    _start_time = position_df.index.get_level_values(level="datetime").min()
    _end_time = position_df.index.get_level_values(level="datetime").max()
    label_data = label_data.loc(axis=0)[:, pd.to_datetime(_start_time) :]
    _result_df = pd.concat([position_df, label_data], axis=1, sort=True).reindex(label_data.index)
    _result_df = _result_df.loc[_result_df.index.get_level_values(1) <= _end_time]
    return _result_df


def _add_bench_to_position(position_df: pd.DataFrame = None, bench: pd.Series = None) -> pd.DataFrame:
    """Concat position with bench

    :param position_df: position DataFrame
    :param bench: report normal data
    :return: concat result
    """
    _temp_df = position_df.reset_index(level="instrument")
    # FIXME: After the stock is bought and sold, the rise and fall of the next trading day are calculated.
    _temp_df["bench"] = bench.shift(-1)
    res_df = _temp_df.set_index(["instrument", _temp_df.index])
    return res_df


def _calculate_label_rank(df: pd.DataFrame) -> pd.DataFrame:
    """calculate label rank

    :param df:
    :return:
    """
    _label_name = "label"

    def _calculate_day_value(g_df: pd.DataFrame):
        g_df = g_df.copy()
        g_df["rank_ratio"] = g_df[_label_name].rank(ascending=False) / len(g_df) * 100

        # Sell: -1, Hold: 0, Buy: 1
        for i in [-1, 0, 1]:
            g_df.loc[g_df["status"] == i, "rank_label_mean"] = g_df[g_df["status"] == i]["rank_ratio"].mean()

        g_df["excess_return"] = g_df[_label_name] - g_df[_label_name].mean()
        return g_df

    return df.groupby(level="datetime").apply(_calculate_day_value)


def get_position_data(
    position: dict,
    label_data: pd.DataFrame,
    report_normal: pd.DataFrame = None,
    calculate_label_rank=False,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Concat position data with pred/report_normal

    :param position: position data
    :param report_normal: report normal, must be container 'bench' column
    :param label_data:
    :param calculate_label_rank:
    :param start_date: start date
    :param end_date: end date
    :return: concat result,
        columns: ['amount', 'cash', 'count', 'price', 'status', 'weight', 'label',
                    'rank_ratio', 'rank_label_mean', 'excess_return', 'score', 'bench']
        index: ['instrument', 'date']
    """
    _position_df = parse_position(position)

    # Add custom_label, rank_ratio, rank_mean, and excess_return field
    _position_df = _add_label_to_position(_position_df, label_data)

    if calculate_label_rank:
        _position_df = _calculate_label_rank(_position_df)

    if report_normal is not None:
        # Add bench field
        _position_df = _add_bench_to_position(_position_df, report_normal["bench"])

    _date_list = _position_df.index.get_level_values(level="datetime")
    start_date = _date_list.min() if start_date is None else start_date
    end_date = _date_list.max() if end_date is None else end_date
    _position_df = _position_df.loc[(start_date <= _date_list) & (_date_list <= end_date)]
    return _position_df
