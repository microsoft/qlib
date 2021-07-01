# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, Text
import numpy as np
import pandas as pd
import copy

from ...log import TimeInspector
from .utils import fetch_df_by_index
from ...utils.serial import Serializable
from ...utils.paral import datetime_groupby_apply

EPS = 1e-12


def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    """
    get a group of columns from multi-index columns DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        with multi of columns.
    group : str
        the name of the feature group, i.e. the first level value of the group index.
    """
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]


class Processor(Serializable):
    def fit(self, df: pd.DataFrame = None):
        """
        learn data processing parameters

        Parameters
        ----------
        df : pd.DataFrame
            When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.

        """
        pass

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        process the data

        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside

        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """
        pass

    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference
        Some processors are not usable for inference.

        Returns
        -------
        bool:
            if it is usable for infenrece.
        """
        return True

    def config(self, **kwargs):
        attr_list = {"fit_start_time", "fit_end_time"}
        for k, v in kwargs.items():
            if k in attr_list and hasattr(self, k):
                setattr(self, k, v)

        for attr in attr_list:
            if attr in kwargs:
                kwargs.pop(attr)
        super().config(**kwargs)


class DropnaProcessor(Processor):
    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        return df.dropna(subset=get_group_columns(df, self.fields_group))


class DropnaLabel(DropnaProcessor):
    def __init__(self, fields_group="label"):
        super().__init__(fields_group=fields_group)

    def is_for_infer(self) -> bool:
        """The samples are dropped according to label. So it is not usable for inference"""
        return False


class DropCol(Processor):
    def __init__(self, col_list=[]):
        self.col_list = col_list

    def __call__(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            mask = df.columns.get_level_values(-1).isin(self.col_list)
        else:
            mask = df.columns.isin(self.col_list)
        return df.loc[:, ~mask]


class FilterCol(Processor):
    def __init__(self, fields_group="feature", col_list=[]):
        self.fields_group = fields_group
        self.col_list = col_list

    def __call__(self, df):

        cols = get_group_columns(df, self.fields_group)
        all_cols = df.columns
        diff_cols = np.setdiff1d(all_cols.get_level_values(-1), cols.get_level_values(-1))
        self.col_list = np.union1d(diff_cols, self.col_list)
        mask = df.columns.get_level_values(-1).isin(self.col_list)
        return df.loc[:, mask]


class TanhProcess(Processor):
    """Use tanh to process noise data"""

    def __call__(self, df):
        def tanh_denoise(data):
            mask = data.columns.get_level_values(1).str.contains("LABEL")
            col = df.columns[~mask]
            data[col] = data[col] - 1
            data[col] = np.tanh(data[col])

            return data

        return tanh_denoise(df)


class ProcessInf(Processor):
    """Process infinity"""

    def __call__(self, df):
        def replace_inf(data):
            def process_inf(df):
                for col in df.columns:
                    # FIXME: Such behavior is very weird
                    df[col] = df[col].replace([np.inf, -np.inf], df[col][~np.isinf(df[col])].mean())
                return df

            data = datetime_groupby_apply(data, process_inf)
            data.sort_index(inplace=True)
            return data

        return replace_inf(df)


class Fillna(Processor):
    """Process NaN"""

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            cols = get_group_columns(df, self.fields_group)
            df.fillna({col: self.fill_value for col in cols}, inplace=True)
        return df


class MinMaxNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.min_val = np.nanmin(df[cols].values, axis=0)
        self.max_val = np.nanmax(df[cols].values, axis=0)
        self.ignore = self.min_val == self.max_val
        self.cols = cols

    def __call__(self, df):
        def normalize(x, min_val=self.min_val, max_val=self.max_val, ignore=self.ignore):
            if (~ignore).all():
                return (x - min_val) / (max_val - min_val)
            for i in range(ignore.size):
                if not ignore[i]:
                    x[i] = (x[i] - min_val) / (max_val - min_val)
            return x

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


class ZScoreNorm(Processor):
    """ZScore Normalization"""

    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.mean_train = np.nanmean(df[cols].values, axis=0)
        self.std_train = np.nanstd(df[cols].values, axis=0)
        self.ignore = self.std_train == 0
        self.cols = cols

    def __call__(self, df):
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train, ignore=self.ignore):
            if (~ignore).all():
                return (x - mean_train) / std_train
            for i in range(ignore.size):
                if not ignore[i]:
                    x[i] = (x[i] - mean_train) / std_train
            return x

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


class RobustZScoreNorm(Processor):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group=None, clip_outlier=True):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier

    def fit(self, df):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        self.cols = get_group_columns(df, self.fields_group)
        X = df[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        self.std_train = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train += EPS
        self.std_train *= 1.4826

    def __call__(self, df):
        X = df[self.cols]
        X -= self.mean_train
        X /= self.std_train
        df[self.cols] = X
        if self.clip_outlier:
            df.clip(-3, 3, inplace=True)
        return df


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        # try not modify original dataframe
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby("datetime").apply(lambda x: (x - x.mean()).div(x.std()))

        return df


class CSRankNorm(Processor):
    """Cross Sectional Rank Normalization"""

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        # try not modify original dataframe
        cols = get_group_columns(df, self.fields_group)
        t = df[cols].groupby("datetime").rank(pct=True)
        t -= 0.5
        t *= 3.46  # NOTE: towards unit std
        df[cols] = t
        return df


class CSZFillna(Processor):
    """Cross Sectional Fill Nan"""

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby("datetime").apply(lambda x: x.fillna(x.mean()))
        return df
