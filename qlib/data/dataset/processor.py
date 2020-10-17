# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import numpy as np
import pandas as pd
import copy

from ...log import TimeInspector
from ...utils.serial import Serializable

EPS = 1e-12


class Processor(Serializable):

    def fit(self, handler, df: pd.DataFrame=None):
        """
        learn data processing parameters

        Parameters
        ----------
        handler : DataHandlerLP
            The data handler to processing data
        df : pd.DataFrame
            When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.
            
        """
        pass

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        process the data

        NOTE: The processor should not change the content of `df`

        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor
        """
        pass


def get_cls_kwargs(processor: [dict, str]) ->  (type, dict):
    """
    extract class and kwargs from processor info

    Parameters
    ----------
    processor : [dict, str]
        similar to processor

    Returns
    -------
    (type, dict):
        the class object and it's arguments.
    """
    if isinstance(processor, dict):
        # raise AttributeError
        klass = globals()[processor['class']]
        kwargs = processor['kwargs']
    elif isinstance(processor, str):
        klass = globals()[processor]
        kwargs = {}
    else:
        raise NotImplementedError(f"This type of input is not supported")
    return klass, kwargs


# Place the function here to be able to reference the Processor
def init_proc_obj(processor: [dict, str, Processor]) -> Processor:
    """
    Initialize Processor Object

    Parameters
    ----------
    processor : [dict, str, Processor]
        The info to initialize processor 

    Returns
    -------
    Processor:
        initialized Processor
    """
    if not isinstance(processor, Processor):
        klass, pkwargs = get_cls_kwargs(processor)
        processor = klass(**pkwargs)
    return processor


class InferProcessor(Processor):
    '''This processor is usable for inference'''
    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference

        Returns
        -------
        bool:
            if it is usable for infenrece
        """
        return True


class NInferProcessor(Processor):
    '''This processor is not usable for inference'''
    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference

        Returns
        -------
        bool:
            if it is usable for infenrece
        """
        return False


class DropnaFeature(InferProcessor):
    def fit(self, handler, df=None):
        self.feature_names = copy.deepcopy(handler.get_feature_names())

    def __call__(self, df):
        return df.dropna(subset=self.feature_names)


class DropnaLabel(InferProcessor):
    def fit(self, handler, df=None):
        self.label_names = copy.deepcopy(handler.get_label_names())

    def __call__(self, df):
        return df.dropna(subset=self.label_names)


class ProcessInf(InferProcessor):
    '''Process infinity  '''
    def __call__(self, df):
        def replace_inf(data):
            def process_inf(df):
                for col in df.columns:
                    # FIXME: Such behavior is very weird
                    df[col] = df[col].replace([np.inf, -np.inf], df[col][~np.isinf(df[col])].mean())
                return df

            data = data.groupby("datetime").apply(process_inf)
            data.sort_index(inplace=True)
            return data
        return replace_inf(df)


class MinMaxNorm(InferProcessor):
    def __init__(self, fit_start_time, fit_end_time):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time

    def fit(self, handler, df):
        # TODO:  看看这里怎么取数据
        self.min_val = np.nanmin(df[handler.get_feature_names()].values, axis=0)
        self.max_val = np.nanmax(df[handler.get_feature_names()].values, axis=0)
        self.ignore = self.min_val == self.max_val
        self.feature_names = copy.deepcopy(handler.get_feature_names())

    def __call__(self, df):
        # FIXME: The df will be changed inplace. It's very dangerous
        # The code below is ugly
        df = df.copy()  # currently copy is used
        def normalize(x, min_val=self.min_val, max_val=self.max_val, ignore=self.ignore):
            if (~ignore).all():
                return (x - min_val) / (max_val - min_val)
            for i in range(ignore.size):
                if not ignore[i]:
                    x[i] = (x[i] - min_val) / (max_val - min_val)
            return x
        df.loc(axis=1)[self.feature_names] = normalize(df[self.feature_names].values)
        return df


class ZscoreNorm(InferProcessor):
    def __init__(self, fit_start_time, fit_end_time):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time

    def fit(self, handler, df):
        self.mean_train = np.nanmean(df[handler.get_feature_names()].values, axis=0)
        self.std_train = np.nanstd(df[handler.get_feature_names()].values, axis=0)
        self.ignore = self.std_train == 0
        self.feature_names = handler.get_feature_names()

    def __call__(self, df):
        # FIXME: The df will be changed inplace. It's very dangerous
        # The code below is ugly
        df = df.copy()  # currently copy is used
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train, ignore=self.ignore):
            if (~ignore).all():
                return (x - mean_train) / std_train
            for i in range(ignore.size):
                if not ignore[i]:
                    x[i] = (x[i] - mean_train) / std_train
            return x
        df.loc(axis=1)[self.feature_names] = normalize(df[self.feature_names].values)
        return df


class ConfigSectionProcessor(InferProcessor):
    def __init__(self, **kwargs):
        super().__init__()
        # Options
        self.fillna_feature = kwargs.get("fillna_feature", True)
        self.fillna_label = kwargs.get("fillna_label", True)
        self.clip_feature_outlier = kwargs.get("clip_feature_outlier", False)
        self.shrink_feature_outlier = kwargs.get("shrink_feature_outlier", True)
        self.clip_label_outlier = kwargs.get("clip_label_outlier", False)

    def fit(self, handler, df=None):
        self.feature_names = handler.get_feature_names()
        self.label_names = handler.get_label_names()

    def __call__(self, df):
        return self._transform(df)

    def _transform(self, df):
        def _label_norm(x):
            x = x - x.mean()  # copy
            x /= x.std()
            if self.clip_label_outlier:
                x.clip(-3, 3, inplace=True)
            if self.fillna_label:
                x.fillna(0, inplace=True)
            return x

        def _feature_norm(x):
            x = x - x.median()  # copy
            x /= x.abs().median() * 1.4826
            if self.clip_feature_outlier:
                x.clip(-3, 3, inplace=True)
            if self.shrink_feature_outlier:
                x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5, inplace=True)
                x.where(x >= -3, -3 - (x + 3).div(x.min() + 3) * 0.5, inplace=True)
            if self.fillna_feature:
                x.fillna(0, inplace=True)
            return x

        TimeInspector.set_time_mark()

        # Copy
        df_new = df.copy()

        # Label
        cols = df.columns[df.columns.str.contains("^LABEL")]
        df_new[cols] = df[cols].groupby(level="datetime").apply(_label_norm)

        # Features
        cols = df.columns[df.columns.str.contains("^KLEN|^KLOW|^KUP")]
        df_new[cols] = df[cols].apply(lambda x: x ** 0.25).groupby(level="datetime").apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^KLOW2|^KUP2")]
        df_new[cols] = df[cols].apply(lambda x: x ** 0.5).groupby(level="datetime").apply(_feature_norm)

        _cols = [
            "KMID",
            "KSFT",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VWAP",
            "ROC",
            "MA",
            "BETA",
            "RESI",
            "QTLU",
            "QTLD",
            "RSV",
            "SUMP",
            "SUMN",
            "SUMD",
            "VSUMP",
            "VSUMN",
            "VSUMD",
        ]
        pat = "|".join(["^" + x for x in _cols])
        cols = df.columns[df.columns.str.contains(pat) & (~df.columns.isin(["HIGH0", "LOW0"]))]
        df_new[cols] = df[cols].groupby(level="datetime").apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")]
        df_new[cols] = df[cols].apply(np.log).groupby(level="datetime").apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^RSQR")]
        df_new[cols] = df[cols].fillna(0).groupby(level="datetime").apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^MAX|^HIGH0")]
        df_new[cols] = df[cols].apply(lambda x: (x - 1) ** 0.5).groupby(level="datetime").apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^MIN|^LOW0")]
        df_new[cols] = df[cols].apply(lambda x: (1 - x) ** 0.5).groupby(level="datetime").apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^CORR|^CORD")]
        df_new[cols] = df[cols].apply(np.exp).groupby(level="datetime").apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^WVMA")]
        df_new[cols] = df[cols].apply(np.log1p).groupby(level="datetime").apply(_feature_norm)

        TimeInspector.log_cost_time("Finished preprocessing data.")

        return df_new
