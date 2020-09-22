# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import numpy as np
import pandas as pd

from ...log import TimeInspector

EPS = 1e-12


class Processor(abc.ABC):
    def __init__(self, feature_names, label_names, **kwargs):
        self.feature_names = feature_names
        self.label_names = label_names

    @abc.abstractmethod
    def __call__(self, df_train, df_valid, df_test):
        pass


class PanelProcessor(Processor):
    """Panel Preprocessor"""

    STD_NORM = "Std"
    MINMAX_NORM = "MinMax"

    def __init__(self, feature_names, label_names, **kwargs):
        super().__init__(feature_names, label_names)
        # Options.
        self.dropna_label = kwargs.get("dropna_label", True)
        self.dropna_feature = kwargs.get("dropna_feature", False)
        self.normalize_method = kwargs.get("normalize_method", None)
        self.replace_inf = kwargs.get("replace_inf_feature", False)

    def __call__(self, df_train, df_valid, df_test):
        """
        Preprocess the data
        :param df:  the dataframe to process data.
        """
        # Drop null labels.
        if self.dropna_label:
            df_train, df_valid, df_test = self._process_drop_null_label(df_train, df_valid, df_test)

        # Dropna if need.
        if self.dropna_feature:
            df_train, df_valid, df_test = self._process_drop_null_feature(df_train, df_valid, df_test)

        # replace the 'inf' with the mean the corresponding dimension
        if self.replace_inf:
            df_train, df_valid, df_test = self._process_replace_inf_feature(df_train, df_valid, df_test)

        # normalize data in given method.
        if self.normalize_method is not None:
            df_train, df_valid, df_test = self._process_normalize_feature(df_train, df_valid, df_test)

        return df_train, df_valid, df_test

    def _process_drop_null_label(self, df_train, df_valid, df_test):
        """
        Drop null labels.
        """
        TimeInspector.set_time_mark()
        df_train = df_train.dropna(subset=self.label_names)
        df_valid = df_valid.dropna(subset=self.label_names)
        # The test data's label is Unkown. They can not be seen when preprocessing
        TimeInspector.log_cost_time("Finished dropping null labels.")

        return df_train, df_valid, df_test

    def _process_drop_null_feature(self, df_train, df_valid, df_test):
        """
        Drop data which contain null features if needed.
        """
        # TODO - `Pandas.dropna` is a low performance method.
        TimeInspector.set_time_mark()
        df_train = df_train.dropna(subset=self.feature_names)
        df_valid = df_valid.dropna(subset=self.feature_names)
        df_test = df_test.dropna(subset=self.feature_names)
        TimeInspector.log_cost_time("Finished dropping nan.")

        return df_train, df_valid, df_test

    def _process_replace_inf_feature(self, df_train, df_valid, df_test):
        """
        replace the 'inf' in feature with the mean of this dimension.
        """
        TimeInspector.set_time_mark()

        def replace_inf(data):
            def process_inf(df):
                for col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], df[col][~np.isinf(df[col])].mean())
                return df

            data = data.groupby("datetime").apply(process_inf)
            data.sort_index(inplace=True)
            return data

        df_train = replace_inf(df_train)
        df_valid = replace_inf(df_valid)
        df_test = replace_inf(df_test)
        TimeInspector.log_cost_time("Finished replace inf.")

        return df_train, df_valid, df_test

    def _process_normalize_feature(self, df_train, df_valid, df_test):
        """
        Normalize data if needed, we provide two method now: min-max normalization and standard normalization.
        """
        TimeInspector.set_time_mark()

        if self.normalize_method == self.MINMAX_NORM:
            min_train = np.nanmin(df_train[self.feature_names].values, axis=0)
            max_train = np.nanmax(df_train[self.feature_names].values, axis=0)
            ignore = min_train == max_train

            def normalize(x, min_train=min_train, max_train=max_train, ignore=ignore):
                if (~ignore).all():
                    return (x - min_train) / (max_train - min_train)
                for i in range(ignore.size):
                    if not ignore[i]:
                        x[i] = (x[i] - min_train) / (max_train - min_train)
                return x

        elif self.normalize_method == self.STD_NORM:
            mean_train = np.nanmean(df_train[self.feature_names].values, axis=0)
            std_train = np.nanstd(df_train[self.feature_names].values, axis=0)
            ignore = std_train == 0

            def normalize(x, mean_train=mean_train, std_train=std_train, ignore=ignore):
                if (~ignore).all():
                    return (x - mean_train) / std_train
                for i in range(ignore.size):
                    if not ignore[i]:
                        x[i] = (x[i] - mean_train) / std_train
                return x

        else:
            raise ValueError("Normalize method {} is not allowed".format(self.normalize_method))

        df_train.loc(axis=1)[self.feature_names] = normalize(df_train[self.feature_names].values)
        df_valid.loc(axis=1)[self.feature_names] = normalize(df_valid[self.feature_names].values)
        df_test.loc(axis=1)[self.feature_names] = normalize(df_test[self.feature_names].values)

        TimeInspector.log_cost_time("Finished normalizing data.")

        return df_train, df_valid, df_test


class ConfigSectionProcessor(Processor):
    def __init__(self, feature_names, label_names, **kwargs):
        super().__init__(feature_names, label_names)
        # Options
        self.fillna_feature = kwargs.get("fillna_feature", True)
        self.fillna_label = kwargs.get("fillna_label", True)
        self.clip_feature_outlier = kwargs.get("clip_feature_outlier", False)
        self.shrink_feature_outlier = kwargs.get("shrink_feature_outlier", True)
        self.clip_label_outlier = kwargs.get("clip_label_outlier", False)

    def __call__(self, *args):
        return [self._transform(x) for x in args]

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
