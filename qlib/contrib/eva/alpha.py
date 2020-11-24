"""
Here is a batch of evaluation functions.

The interface should be redesigned carefully in the future.
"""
import pandas as pd

from typing import Tuple


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
