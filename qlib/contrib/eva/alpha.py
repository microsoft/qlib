"""
Here is a batch of evaluation functions.

The interface should be redesigned carefully in the future.
"""

import pandas as pd
from typing import Tuple
from qlib import get_module_logger
from qlib.utils.paral import complex_parallel, DelayedDict
from joblib import Parallel, delayed


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
        label = label - label.groupby(level=date_col, group_keys=False).mean()
    if int(1 / quantile) >= len(label.index.get_level_values(1).unique()):
        raise ValueError("Need more instruments to calculate precision")

    df = pd.DataFrame({"pred": pred, "label": label})
    if dropna:
        df.dropna(inplace=True)

    group = df.groupby(level=date_col, group_keys=False)

    def N(x):
        return int(len(x) * quantile)

    # find the top/low quantile of prediction and treat them as long and short target
    long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label)
    short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label)

    groupll = long.groupby(date_col, group_keys=False)
    l_dom = groupll.apply(lambda x: x > 0)
    l_c = groupll.count()

    groups = short.groupby(date_col, group_keys=False)
    s_dom = groups.apply(lambda x: x < 0)
    s_c = groups.count()
    return (l_dom.groupby(date_col, group_keys=False).sum() / l_c), (
        s_dom.groupby(date_col, group_keys=False).sum() / s_c
    )


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
    group = df.groupby(level=date_col, group_keys=False)

    def N(x):
        return int(len(x) * quantile)

    r_long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label.mean())
    r_short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label.mean())
    r_avg = group.label.mean()
    return (r_long - r_short) / 2, r_avg


def calc_group_returns(pred_label: pd.DataFrame, n_groups: int = 5, date_col: str = "datetime") -> pd.DataFrame:
    """Mean forward return of score groups.

    ``pred_label`` has columns ``score`` and ``label`` and a MultiIndex that
    includes ``date_col``. Inside each date, names are ordered by ``score``
    from high to low. Each group holds ``len(day) // n_groups`` names, and the
    last group also keeps the remainder. ``Group1`` is the highest-score
    bucket. ``long-short`` is then a mean of that first bucket against a
    slightly larger last bucket whenever the cross-section does not divide
    evenly. A day with fewer names than ``n_groups`` has no full bucket, so
    every group is NaN.

    Names with a missing score are left out of the groups. ``long-average``
    still subtracts the mean label of every name on that date.

    Parameters
    ----------
    pred_label : pd.DataFrame
        Columns ``score`` and ``label``.
    n_groups : int
        Number of score buckets. Must be at least 2.
    date_col : str
        Datetime level name.

    Returns
    -------
    pd.DataFrame
        Columns ``Group1`` ... ``Group{n_groups}``, ``long-short``, and
        ``long-average``, indexed by date.
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2, got {n_groups}")
    if "score" not in pred_label.columns or "label" not in pred_label.columns:
        raise KeyError("pred_label must contain 'score' and 'label' columns")

    ranked = pred_label.dropna(subset=["score"]).sort_values("score", ascending=False)

    def group_means(day: pd.DataFrame) -> pd.Series:
        size = len(day) // n_groups
        means = {}
        for i in range(n_groups):
            name = f"Group{i + 1}"
            if size == 0:
                means[name] = float("nan")
                continue
            start = size * i
            # The bottom bucket keeps names that do not fill another full group.
            end = len(day) if i == n_groups - 1 else size * (i + 1)
            means[name] = day["label"].iloc[start:end].mean()
        return pd.Series(means)

    grouped = ranked.groupby(level=date_col, group_keys=False).apply(group_means)
    if isinstance(grouped, pd.Series):
        grouped = grouped.unstack()
    last = f"Group{n_groups}"
    grouped["long-short"] = grouped["Group1"] - grouped[last]
    baseline = pred_label.groupby(level=date_col, group_keys=False)["label"].mean()
    grouped["long-average"] = grouped["Group1"] - baseline
    return grouped


def pred_autocorr(pred: pd.Series, lag=1, inst_col="instrument", date_col="datetime"):
    """pred_autocorr.

    Limitation:
    - If the datetime is not sequential densely, the correlation will be calulated based on adjacent dates. (some users may expected NaN)

    :param pred: pd.Series with following format
                instrument  datetime
                SH600000    2016-01-04   -0.000403
                            2016-01-05   -0.000753
                            2016-01-06   -0.021801
                            2016-01-07   -0.065230
                            2016-01-08   -0.062465
    :type pred: pd.Series
    :param lag:
    """
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
        get_module_logger("pred_autocorr").warning(f"Only the first column in {pred.columns} of `pred` is kept")
    pred_ustk = pred.sort_index().unstack(inst_col)
    corr_s = {}
    for (idx, cur), (_, prev) in zip(pred_ustk.iterrows(), pred_ustk.shift(lag).iterrows()):
        corr_s[idx] = cur.corr(prev)
    corr_s = pd.Series(corr_s).sort_index()
    return corr_s


def pred_autocorr_all(pred_dict, n_jobs=-1, **kwargs):
    """
    calculate auto correlation for pred_dict

    Parameters
    ----------
    pred_dict : dict
        A dict like {<method_name>:  <prediction>}
    kwargs :
        all these arguments will be passed into pred_autocorr
    """
    ac_dict = {}
    for k, pred in pred_dict.items():
        ac_dict[k] = delayed(pred_autocorr)(pred, **kwargs)
    return complex_parallel(Parallel(n_jobs=n_jobs, verbose=10), ac_dict)


def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False) -> (pd.Series, pd.Series):
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
    ic = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"]))
    ric = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
    if dropna:
        return ic.dropna(), ric.dropna()
    else:
        return ic, ric


def calc_all_ic(pred_dict_all, label, date_col="datetime", dropna=False, n_jobs=-1):
    """calc_all_ic.

    Parameters
    ----------
    pred_dict_all :
        A dict like {<method_name>:  <prediction>}
    label:
        A pd.Series of label values

    Returns
    -------
    {'Q2+IND_z': {'ic': <ic series like>
                          2016-01-04   -0.057407
                          ...
                          2020-05-28    0.183470
                          2020-05-29    0.171393
                  'ric': <rank ic series like>
                          2016-01-04   -0.040888
                          ...
                          2020-05-28    0.236665
                          2020-05-29    0.183886
                  }
    ...}
    """
    pred_all_ics = {}
    for k, pred in pred_dict_all.items():
        pred_all_ics[k] = DelayedDict(["ic", "ric"], delayed(calc_ic)(pred, label, date_col=date_col, dropna=dropna))
    pred_all_ics = complex_parallel(Parallel(n_jobs=n_jobs, verbose=10), pred_all_ics)
    return pred_all_ics
