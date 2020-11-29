# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from joblib import Parallel, delayed
import pandas as pd


def datetime_groupby_apply(df, apply_func, axis=0, level="datetime", resample_rule="M", n_jobs=-1, skip_group=False):
    """datetime_groupby_apply
    This function will apply the `apply_func` on the datetime level index.

    Parameters
    ----------
    df :
        DataFrame for processing
    apply_func :
        apply_func for processing the data
    axis :
        which axis is the datetime level located
    level :
        which level is the datetime level
    resample_rule :
        How to resample the data to calculating parallel
    n_jobs :
        n_jobs for joblib
    Returns:
        pd.DataFrame
    """

    def _naive_group_apply(df):
        return df.groupby(axis=axis, level=level).apply(apply_func)

    if n_jobs != 1:
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(_naive_group_apply)(sub_df) for idx, sub_df in df.resample(resample_rule, axis=axis, level=level)
        )
        return pd.concat(dfs, axis=axis).sort_index()
    else:
        return _naive_group_apply(df)
