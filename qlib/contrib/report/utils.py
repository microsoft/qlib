# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt


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
