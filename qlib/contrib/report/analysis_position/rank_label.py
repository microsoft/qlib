# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from typing import Iterable

import pandas as pd
import plotly.graph_objs as go

from ..graph import ScatterGraph
from ..analysis_position.parse_position import get_position_data


def _get_figure_with_position(
    position: dict, label_data: pd.DataFrame, start_date=None, end_date=None
) -> Iterable[go.Figure]:
    """Get average analysis figures

    :param position: position
    :param label_data:
    :param start_date:
    :param end_date:
    :return:
    """
    _position_df = get_position_data(
        position,
        label_data,
        calculate_label_rank=True,
        start_date=start_date,
        end_date=end_date,
    )

    res_dict = dict()
    _pos_gp = _position_df.groupby(level=1)
    for _item in _pos_gp:
        _date = _item[0]
        _day_df = _item[1]

        _day_value = res_dict.setdefault(_date, {})
        for _i, _name in {0: "Hold", 1: "Buy", -1: "Sell"}.items():
            _temp_df = _day_df[_day_df["status"] == _i]
            if _temp_df.empty:
                _day_value[_name] = 0
            else:
                _day_value[_name] = _temp_df["rank_label_mean"].values[0]

    _res_df = pd.DataFrame.from_dict(res_dict, orient="index")
    # FIXME: support HIGH-FREQ
    _res_df.index = _res_df.index.strftime("%Y-%m-%d")
    for _col in _res_df.columns:
        yield ScatterGraph(
            _res_df.loc[:, [_col]],
            layout=dict(
                title=_col,
                xaxis=dict(type="category", tickangle=45),
                yaxis=dict(title="lable-rank-ratio: %"),
            ),
            graph_kwargs=dict(mode="lines+markers"),
        ).figure


def rank_label_graph(
    position: dict,
    label_data: pd.DataFrame,
    start_date=None,
    end_date=None,
    show_notebook=True,
) -> Iterable[go.Figure]:
    """Ranking percentage of stocks buy, sell, and holding on the trading day.
    Average rank-ratio(similar to **sell_df['label'].rank(ascending=False) / len(sell_df)**) of daily trading

        Example:


            .. code-block:: python

                from qlib.data import D
                from qlib.contrib.evaluate import backtest
                from qlib.contrib.strategy import TopkDropoutStrategy

                # backtest parameters
                bparas = {}
                bparas['limit_threshold'] = 0.095
                bparas['account'] = 1000000000

                sparas = {}
                sparas['topk'] = 50
                sparas['n_drop'] = 230
                strategy = TopkDropoutStrategy(**sparas)

                _, positions = backtest(pred_df, strategy, **bparas)

                pred_df_dates = pred_df.index.get_level_values(level='datetime')
                features_df = D.features(D.instruments('csi500'), ['Ref($close, -1)/$close-1'], pred_df_dates.min(), pred_df_dates.max())
                features_df.columns = ['label']

                qcr.analysis_position.rank_label_graph(positions, features_df, pred_df_dates.min(), pred_df_dates.max())


    :param position: position data; **qlib.backtest.backtest** result.
    :param label_data: **D.features** result; index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[label]**.
    **The label T is the change from T to T+1**, it is recommended to use ``close``, example: `D.features(D.instruments('csi500'), ['Ref($close, -1)/$close-1'])`.


            .. code-block:: python

                                                label
                instrument  datetime
                SH600004        2017-12-11  -0.013502
                                2017-12-12  -0.072367
                                2017-12-13  -0.068605
                                2017-12-14  0.012440
                                2017-12-15  -0.102778


    :param start_date: start date
    :param end_date: end_date
    :param show_notebook: **True** or **False**. If True, show graph in notebook, else return figures.
    :return:
    """
    position = copy.deepcopy(position)
    label_data.columns = ["label"]
    _figures = _get_figure_with_position(position, label_data, start_date, end_date)
    if show_notebook:
        ScatterGraph.show_graph_in_notebook(_figures)
    else:
        return _figures
