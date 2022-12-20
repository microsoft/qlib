# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import pandas as pd


def sub_fig_generator(sub_fs=(3, 3), col_n=10, row_n=1, wspace=None, hspace=None, sharex=False, sharey=False):
    """sub_fig_generator.
    it will return a generator, each row contains <col_n> sub graph

    FIXME: Known limitation:
    - The last row will not be plotted automatically, please plot it outside the function

    Parameters
    ----------
    sub_fs :
        the figure size of each subgraph in <col_n> * <row_n> subgraphs
    col_n :
        the number of subgraph in each row;  It will generating a new graph after generating <col_n> of subgraphs.
    row_n :
        the number of subgraph in each column
    wspace :
        the width of the space for subgraphs in each row
    hspace :
        the height of blank space for subgraphs in each column
        You can try 0.3 if you feel it is too crowded

    Returns
    -------
    It will return graphs with the shape of <col_n> each iter (it is squeezed).
    """
    assert col_n > 1

    while True:
        fig, axes = plt.subplots(
            row_n, col_n, figsize=(sub_fs[0] * col_n, sub_fs[1] * row_n), sharex=sharex, sharey=sharey
        )
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        axes = axes.reshape(row_n, col_n)

        for col in range(col_n):
            res = axes[:, col].squeeze()
            if res.size == 1:
                res = res.item()
            yield res
        plt.show()


def guess_plotly_rangebreaks(dt_index: pd.DatetimeIndex):
    """
    This function `guesses` the rangebreaks required to remove gaps in datetime index.
    It basically calculates the difference between a `continuous` datetime index and index given.

    For more details on `rangebreaks` params in plotly, see
    https://plotly.com/python/reference/layout/xaxis/#layout-xaxis-rangebreaks

    Parameters
    ----------
    dt_index: pd.DatetimeIndex
    The datetimes of the data.

    Returns
    -------
    the `rangebreaks` to be passed into plotly axis.

    """
    dt_idx = dt_index.sort_values()
    gaps = dt_idx[1:] - dt_idx[:-1]
    min_gap = gaps.min()
    gaps_to_break = {}
    for gap, d in zip(gaps, dt_idx[:-1]):
        if gap > min_gap:
            gaps_to_break.setdefault(gap - min_gap, []).append(d + min_gap)
    return [dict(values=v, dvalue=int(k.total_seconds() * 1000)) for k, v in gaps_to_break.items()]
