# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from typing import Iterable

import pandas as pd
import plotly.graph_objs as go

from ..graph import BaseGraph, SubplotsGraph

from ..analysis_position.parse_position import get_position_data


def _get_cum_return_data_with_position(
    position: dict,
    report_normal: pd.DataFrame,
    label_data: pd.DataFrame,
    start_date=None,
    end_date=None,
):
    """

    :param position:
    :param report_normal:
    :param label_data:
    :param start_date:
    :param end_date:
    :return:
    """
    _cumulative_return_df = get_position_data(
        position=position,
        report_normal=report_normal,
        label_data=label_data,
        start_date=start_date,
        end_date=end_date,
    ).copy()

    _cumulative_return_df["label"] = _cumulative_return_df["label"] - _cumulative_return_df["bench"]
    _cumulative_return_df = _cumulative_return_df.dropna()
    df_gp = _cumulative_return_df.groupby(level="datetime")
    result_list = []
    for gp in df_gp:
        date = gp[0]
        day_df = gp[1]

        _hold_df = day_df[day_df["status"] == 0]
        _buy_df = day_df[day_df["status"] == 1]
        _sell_df = day_df[day_df["status"] == -1]

        hold_value = (_hold_df["label"] * _hold_df["weight"]).sum()
        hold_weight = _hold_df["weight"].sum()
        hold_mean = (hold_value / hold_weight) if hold_weight else 0

        sell_value = (_sell_df["label"] * _sell_df["weight"]).sum()
        sell_weight = _sell_df["weight"].sum()
        sell_mean = (sell_value / sell_weight) if sell_weight else 0

        buy_value = (_buy_df["label"] * _buy_df["weight"]).sum()
        buy_weight = _buy_df["weight"].sum()
        buy_mean = (buy_value / buy_weight) if buy_weight else 0

        result_list.append(
            dict(
                hold_value=hold_value,
                hold_mean=hold_mean,
                hold_weight=hold_weight,
                buy_value=buy_value,
                buy_mean=buy_mean,
                buy_weight=buy_weight,
                sell_value=sell_value,
                sell_mean=sell_mean,
                sell_weight=sell_weight,
                buy_minus_sell_value=buy_value - sell_value,
                buy_minus_sell_mean=buy_mean - sell_mean,
                buy_plus_sell_weight=buy_weight + sell_weight,
                date=date,
            )
        )

    r_df = pd.DataFrame(data=result_list)
    r_df["cum_hold"] = r_df["hold_mean"].cumsum()
    r_df["cum_buy"] = r_df["buy_mean"].cumsum()
    r_df["cum_sell"] = r_df["sell_mean"].cumsum()
    r_df["cum_buy_minus_sell"] = r_df["buy_minus_sell_mean"].cumsum()
    return r_df


def _get_figure_with_position(
    position: dict,
    report_normal: pd.DataFrame,
    label_data: pd.DataFrame,
    start_date=None,
    end_date=None,
) -> Iterable[go.Figure]:
    """Get average analysis figures

    :param position: position
    :param report_normal:
    :param label_data:
    :param start_date:
    :param end_date:
    :return:
    """

    cum_return_df = _get_cum_return_data_with_position(position, report_normal, label_data, start_date, end_date)
    cum_return_df = cum_return_df.set_index("date")
    # FIXME: support HIGH-FREQ
    cum_return_df.index = cum_return_df.index.strftime("%Y-%m-%d")

    # Create figures
    for _t_name in ["buy", "sell", "buy_minus_sell", "hold"]:
        sub_graph_data = [
            (
                "cum_{}".format(_t_name),
                dict(row=1, col=1, graph_kwargs={"mode": "lines+markers", "xaxis": "x3"}),
            ),
            (
                "{}_weight".format(_t_name.replace("minus", "plus") if "minus" in _t_name else _t_name),
                dict(row=2, col=1),
            ),
            (
                "{}_value".format(_t_name),
                dict(row=1, col=2, kind="HistogramGraph", graph_kwargs={}),
            ),
        ]

        _default_xaxis = dict(showline=False, zeroline=True, tickangle=45)
        _default_yaxis = dict(zeroline=True, showline=True, showticklabels=True)
        sub_graph_layout = dict(
            xaxis1=dict(**_default_xaxis, type="category", showticklabels=False),
            xaxis3=dict(**_default_xaxis, type="category"),
            xaxis2=_default_xaxis,
            yaxis1=dict(**_default_yaxis, title=_t_name),
            yaxis2=_default_yaxis,
            yaxis3=_default_yaxis,
        )

        mean_value = cum_return_df["{}_value".format(_t_name)].mean()
        layout = dict(
            height=500,
            title=f"{_t_name}(the red line in the histogram on the right represents the average)",
            shapes=[
                {
                    "type": "line",
                    "xref": "x2",
                    "yref": "paper",
                    "x0": mean_value,
                    "y0": 0,
                    "x1": mean_value,
                    "y1": 1,
                    # NOTE: 'fillcolor': '#d3d3d3', 'opacity': 0.3,
                    "line": {"color": "red", "width": 1},
                },
            ],
        )

        kind_map = dict(kind="ScatterGraph", kwargs=dict(mode="lines+markers"))
        specs = [
            [{"rowspan": 1}, {"rowspan": 2}],
            [{"rowspan": 1}, None],
        ]
        subplots_kwargs = dict(
            vertical_spacing=0.01,
            rows=2,
            cols=2,
            row_width=[1, 2],
            column_width=[3, 1],
            print_grid=False,
            specs=specs,
        )
        yield SubplotsGraph(
            cum_return_df,
            layout=layout,
            kind_map=kind_map,
            sub_graph_layout=sub_graph_layout,
            sub_graph_data=sub_graph_data,
            subplots_kwargs=subplots_kwargs,
        ).figure


def cumulative_return_graph(
    position: dict,
    report_normal: pd.DataFrame,
    label_data: pd.DataFrame,
    show_notebook=True,
    start_date=None,
    end_date=None,
) -> Iterable[go.Figure]:
    """Backtest buy, sell, and holding cumulative return graph

        Example:


            .. code-block:: python

                from qlib.data import D
                from qlib.contrib.evaluate import risk_analysis, backtest, long_short_backtest
                from qlib.contrib.strategy import TopkDropoutStrategy

                # backtest parameters
                bparas = {}
                bparas['limit_threshold'] = 0.095
                bparas['account'] = 1000000000

                sparas = {}
                sparas['topk'] = 50
                sparas['n_drop'] = 5
                strategy = TopkDropoutStrategy(**sparas)

                report_normal_df, positions = backtest(pred_df, strategy, **bparas)

                pred_df_dates = pred_df.index.get_level_values(level='datetime')
                features_df = D.features(D.instruments('csi500'), ['Ref($close, -1)/$close - 1'], pred_df_dates.min(), pred_df_dates.max())
                features_df.columns = ['label']

                qcr.analysis_position.cumulative_return_graph(positions, report_normal_df, features_df)


        Graph desc:

            - Axis X: Trading day.
            - Axis Y:
            - Above axis Y: `(((Ref($close, -1)/$close - 1) * weight).sum() / weight.sum()).cumsum()`.
            - Below axis Y: Daily weight sum.
            - In the **sell** graph, `y < 0` stands for profit; in other cases, `y > 0` stands for profit.
            - In the **buy_minus_sell** graph, the **y** value of the **weight** graph at the bottom is `buy_weight + sell_weight`.
            - In each graph, the **red line** in the histogram on the right represents the average.

    :param position: position data
    :param report_normal:


            .. code-block:: python

                                return      cost        bench       turnover
                date
                2017-01-04  0.003421    0.000864    0.011693    0.576325
                2017-01-05  0.000508    0.000447    0.000721    0.227882
                2017-01-06  -0.003321   0.000212    -0.004322   0.102765
                2017-01-09  0.006753    0.000212    0.006874    0.105864
                2017-01-10  -0.000416   0.000440    -0.003350   0.208396


    :param label_data: `D.features` result; index is `pd.MultiIndex`, index name is [`instrument`, `datetime`]; columns names is [`label`].

        **The label T is the change from T to T+1**, it is recommended to use ``close``, example: `D.features(D.instruments('csi500'), ['Ref($close, -1)/$close-1'])`


            .. code-block:: python

                                                label
                instrument  datetime
                SH600004        2017-12-11  -0.013502
                                2017-12-12  -0.072367
                                2017-12-13  -0.068605
                                2017-12-14  0.012440
                                2017-12-15  -0.102778


    :param show_notebook: True or False. If True, show graph in notebook, else return figures
    :param start_date: start date
    :param end_date: end date
    :return:
    """
    position = copy.deepcopy(position)
    report_normal = report_normal.copy()
    label_data.columns = ["label"]
    _figures = _get_figure_with_position(position, report_normal, label_data, start_date, end_date)
    if show_notebook:
        BaseGraph.show_graph_in_notebook(_figures)
    else:
        return _figures
