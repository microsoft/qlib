# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import numpy as np
import pandas as pd
import copy

from ...log import TimeInspector
from ...utils.serial import Serializable
from ...utils.paral import datetime_groupby_apply

EPS = 1e-12


def get_group_columns(df: pd.DataFrame, group: str):
    """
    get a group of columns from multi-index columns DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        with multi of columns
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
            The raw_df of handler or result from previous processor
        """
        pass

    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference
        Some processors are not usable for inference.

        Returns
        -------
        bool:
            if it is usable for infenrece
        """
        return True


class DropnaProcessor(Processor):
    def __init__(self, group=None):
        self.group = group

    def __call__(self, df):
        return df.dropna(subset=get_group_columns(df, self.group))


class DropnaLabel(DropnaProcessor):
    def __init__(self, group="label"):
        super().__init__(group=group)

    def is_for_infer(self) -> bool:
        """The samples are dropped according to label. So it is not usable for inference"""
        return False


class ProcessInf(Processor):
    """Process infinity  """

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


class MinMaxNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # FIXME: time is not used
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df):
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


class ZscoreNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # FIXME: time is not used
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df):
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


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        # try not modify original dataframe
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby("datetime").apply(lambda df: (df - df.mean()).div(df.std()))
        return df
