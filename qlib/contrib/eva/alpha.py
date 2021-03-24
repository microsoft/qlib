"""
Here is a batch of evaluation functions.

The interface should be redesigned carefully in the future.
"""
import pandas as pd

from typing import Tuple


def calc_long_short_prec(
    pred: pd.Series, label: pd.Series, date_col="datetime", quantile: float = 0.2, dropna=False, is_alpha=False
) -> Tuple[pd.Series, pd.Series]:
    """
    calculate the precision for long and short operation


    :param pred/label: index is **pd.MultiIndex**, index name is **[datetime, instruments]**; columns names is **[score]**.

            .. code-block:: python
                                                  score
                datetime            instrument
                2020-12-01 09:30:00 SH600068    0.553634
                                    SH600195    0.550017
                                    SH600276    0.540321
                                    SH600584    0.517297
                                    SH600715    0.544674
    label :
        label
    date_col :
        date_col

    Returns
    -------
    (pd.Series, pd.Series)
        long precision and short precision in time level
    """
    if is_alpha:
        label = label - label.mean(level=date_col)
    if int(1 / quantile) >= len(label.index.get_level_values(1).unique()):
        raise ValueError("Need more instruments to calculate precision")

    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)

    group = df.groupby(level=date_col)

    N = lambda x: int(len(x) * quantile)
    # find the top/low quantile of prediction and treat them as long and short target
    long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label).reset_index(level=0, drop=True)
    short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label).reset_index(level=0, drop=True)

    groupll = long.groupby(date_col)
    l_dom = groupll.apply(lambda x: x > 0)
    l_c = groupll.count()

    groups = short.groupby(date_col)
    s_dom = groups.apply(lambda x: x < 0)
    s_c = groups.count()
    return (l_dom.groupby(date_col).sum() / l_c), (s_dom.groupby(date_col).sum() / s_c)


def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False) -> Tuple[pd.Series, pd.Series]:
    """calc_ic.

    Parameters
    ----------
    pred :
        pred
    label :
        label
    date_col :
        date_col

    Returns
    -------
    (pd.Series, pd.Series)
        ic and rank ic
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    ic = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"]))
    ric = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
    if dropna:
        return ic.dropna(), ric.dropna()
    else:
        return ic, ric


def calc_long_short_return(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    quantile: float = 0.2,
    dropna: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    calculate long-short return

    Note:
        `label` must be raw stock returns.

    Parameters
    ----------
    pred : pd.Series
        stock predictions
    label : pd.Series
        stock returns
    date_col : str
        datetime index name
    quantile : float
        long-short quantile

    Returns
    ----------
    long_short_r : pd.Series
        daily long-short returns
    long_avg_r : pd.Series
        daily long-average returns
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)
    group = df.groupby(level=date_col)
    N = lambda x: int(len(x) * quantile)
    r_long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label.mean())
    r_short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label.mean())
    r_avg = group.label.mean()
    return (r_long - r_short) / 2, r_avg
