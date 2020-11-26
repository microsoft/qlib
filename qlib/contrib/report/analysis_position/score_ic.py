# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd

from ..graph import ScatterGraph


def _get_score_ic(pred_label: pd.DataFrame):
    """

    :param pred_label:
    :return:
    """
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how="any", inplace=True)
    _ic = concat_data.groupby(level="datetime").apply(lambda x: x["label"].corr(x["score"]))
    _rank_ic = concat_data.groupby(level="datetime").apply(lambda x: x["label"].corr(x["score"], method="spearman"))
    return pd.DataFrame({"ic": _ic, "rank_ic": _rank_ic})


def score_ic_graph(pred_label: pd.DataFrame, show_notebook: bool = True) -> [list, tuple]:
    """score IC

        Example:


            .. code-block:: python

                from qlib.data import D
                from qlib.contrib.report import analysis_position
                pred_df_dates = pred_df.index.get_level_values(level='datetime')
                features_df = D.features(D.instruments('csi500'), ['Ref($close, -2)/Ref($close, -1)-1'], pred_df_dates.min(), pred_df_dates.max())
                features_df.columns = ['label']
                pred_label = pd.concat([features_df, pred], axis=1, sort=True).reindex(features_df.index)
                analysis_position.score_ic_graph(pred_label)


    :param pred_label: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score, label]**.


            .. code-block:: python

                instrument  datetime        score         label
                SH600004  2017-12-11     -0.013502       -0.013502
                            2017-12-12   -0.072367       -0.072367
                            2017-12-13   -0.068605       -0.068605
                            2017-12-14    0.012440        0.012440
                            2017-12-15   -0.102778       -0.102778


    :param show_notebook: whether to display graphics in notebook, the default is **True**.
    :return: if show_notebook is True, display in notebook; else return **plotly.graph_objs.Figure** list.
    """
    _ic_df = _get_score_ic(pred_label)
    # FIXME: support HIGH-FREQ
    _ic_df.index = _ic_df.index.strftime("%Y-%m-%d")
    _figure = ScatterGraph(
        _ic_df,
        layout=dict(title="Score IC", xaxis=dict(type="category", tickangle=45)),
        graph_kwargs={"mode": "lines+markers"},
    ).figure
    if show_notebook:
        ScatterGraph.show_graph_in_notebook([_figure])
    else:
        return (_figure,)
