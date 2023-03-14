# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, Text, Optional
import numpy as np
import pandas as pd

from qlib.utils.data import robust_zscore, zscore
from ...constant import EPS
from .utils import fetch_df_by_index
from ...utils.serial import Serializable
from ...utils.paral import datetime_groupby_apply
from qlib.data.inst_processor import InstProcessor
from qlib.data import D


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

    def readonly(self) -> bool:
        """
        Does the processor treat the input data readonly (i.e. does not write the input data) when processing

        Knowning the readonly information is helpful to the Handler to avoid uncessary copy
        """
        return False

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

    def readonly(self):
        return True


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

    def readonly(self):
        return True


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

    def readonly(self):
        return True


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
            # this implementation is extremely slow
            # df.fillna({col: self.fill_value for col in cols}, inplace=True)

            # So we use numpy to accelerate filling values
            nan_select = np.isnan(df.values)
            nan_select[:, ~df.columns.isin(cols)] = False
            df.values[nan_select] = self.fill_value
        return df


class MinMaxNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.min_val = np.nanmin(df[cols].values, axis=0)
        self.max_val = np.nanmax(df[cols].values, axis=0)
        self.ignore = self.min_val == self.max_val
        # To improve the speed, we set the value of `min_val` to `0` for the columns that do not need to be processed,
        # and the value of `max_val` to `1`, when using `(x - min_val) / (max_val - min_val)` for uniform calculation,
        # the columns that do not need to be processed will be calculated by `(x - 0) / (1 - 0)`,
        # as you can see, the columns that do not need to be processed, will not be affected.
        for _i, _con in enumerate(self.ignore):
            if _con:
                self.min_val[_i] = 0
                self.max_val[_i] = 1
        self.cols = cols

    def __call__(self, df):
        def normalize(x, min_val=self.min_val, max_val=self.max_val):
            return (x - min_val) / (max_val - min_val)

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


class ZScoreNorm(Processor):
    """ZScore Normalization"""

    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.mean_train = np.nanmean(df[cols].values, axis=0)
        self.std_train = np.nanstd(df[cols].values, axis=0)
        self.ignore = self.std_train == 0
        # To improve the speed, we set the value of `std_train` to `1` for the columns that do not need to be processed,
        # and the value of `mean_train` to `0`, when using `(x - mean_train) / std_train` for uniform calculation,
        # the columns that do not need to be processed will be calculated by `(x - 0) / 1`,
        # as you can see, the columns that do not need to be processed, will not be affected.
        for _i, _con in enumerate(self.ignore):
            if _con:
                self.std_train[_i] = 1
                self.mean_train[_i] = 0
        self.cols = cols

    def __call__(self, df):
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train):
            return (x - mean_train) / std_train

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
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier

    def fit(self, df: pd.DataFrame = None):
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
        if self.clip_outlier:
            X = np.clip(X, -3, 3)
        df[self.cols] = X
        return df


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None, method="zscore"):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def __call__(self, df):
        # try not modify original dataframe
        if not isinstance(self.fields_group, list):
            self.fields_group = [self.fields_group]
        for g in self.fields_group:
            cols = get_group_columns(df, g)
            df[cols] = df[cols].groupby("datetime", group_keys=False).apply(self.zscore_func)
        return df


class CSRankNorm(Processor):
    """
    Cross Sectional Rank Normalization.
    "Cross Sectional" is often used to describe data operations.
    The operations across different stocks are often called Cross Sectional Operation.

    For example, CSRankNorm is an operation that grouping the data by each day and rank `across` all the stocks in each day.

    Explanation about 3.46 & 0.5

    .. code-block:: python

        import numpy as np
        import pandas as pd
        x = np.random.random(10000)  # for any variable
        x_rank = pd.Series(x).rank(pct=True)  # if it is converted to rank, it will be a uniform distributed
        x_rank_norm = (x_rank - x_rank.mean()) / x_rank.std()  # Normally, we will normalize it to make it like normal distribution

        x_rank.mean()   # accounts for 0.5
        1 / x_rank.std()  # accounts for 3.46

    """

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
        df[cols] = df[cols].groupby("datetime", group_keys=False).apply(lambda x: x.fillna(x.mean()))
        return df


class HashStockFormat(Processor):
    """Process the storage of from df into hasing stock format"""

    def __call__(self, df: pd.DataFrame):
        from .storage import HashingStockStorage  # pylint: disable=C0415

        return HashingStockStorage.from_df(df)


class TimeRangeFlt(InstProcessor):
    """
    This is a filter to filter stock.
    Only keep the data that exist from start_time to end_time (the existence in the middle is not checked.)
    WARNING:  It may induce leakage!!!
    """

    def __init__(
        self,
        start_time: Optional[Union[pd.Timestamp, str]] = None,
        end_time: Optional[Union[pd.Timestamp, str]] = None,
        freq: str = "day",
    ):
        """
        Parameters
        ----------
        start_time : Optional[Union[pd.Timestamp, str]]
            The data must start earlier (or equal) than `start_time`
            None indicates data will not be filtered based on `start_time`
        end_time : Optional[Union[pd.Timestamp, str]]
            similar to start_time
        freq : str
            The frequency of the calendar
        """
        # Align to calendar before filtering
        cal = D.calendar(start_time=start_time, end_time=end_time, freq=freq)
        self.start_time = None if start_time is None else cal[0]
        self.end_time = None if end_time is None else cal[-1]

    def __call__(self, df: pd.DataFrame, instrument, *args, **kwargs):
        if (
            df.empty
            or (self.start_time is None or df.index.min() <= self.start_time)
            and (self.end_time is None or df.index.max() >= self.end_time)
        ):
            return df
        return df.head(0)
