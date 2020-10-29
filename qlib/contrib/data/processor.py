import numpy as np
import pandas as pd
import copy

from ...log import TimeInspector
from ...utils.serial import Serializable
from ...data.dataset.processor import Processor, get_group_columns


class ConfigSectionProcessor(Processor):
    """
    This processor is designed for Alpha158. And will be replaced by simple processors in the future
    """

    def __init__(self, fields_group=None, **kwargs):
        super().__init__()
        # Options
        self.fillna_feature = kwargs.get("fillna_feature", True)
        self.fillna_label = kwargs.get("fillna_label", True)
        self.clip_feature_outlier = kwargs.get("clip_feature_outlier", False)
        self.shrink_feature_outlier = kwargs.get("shrink_feature_outlier", True)
        self.clip_label_outlier = kwargs.get("clip_label_outlier", False)

        self.fields_group = None

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

        # Copy the focus part and change it to single level
        selected_cols = get_group_columns(df, self.fields_group)
        df_focus = df[selected_cols].copy()
        if len(df_focus.columns.levels) > 1:
            df_focus = df_focus.droplevel(level=0)

        # Label
        cols = df_focus.columns[df_focus.columns.str.contains("^LABEL")]
        df_focus[cols] = df_focus[cols].groupby(level="datetime").apply(_label_norm)

        # Features
        cols = df_focus.columns[df_focus.columns.str.contains("^KLEN|^KLOW|^KUP")]
        df_focus[cols] = df_focus[cols].apply(lambda x: x ** 0.25).groupby(level="datetime").apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^KLOW2|^KUP2")]
        df_focus[cols] = df_focus[cols].apply(lambda x: x ** 0.5).groupby(level="datetime").apply(_feature_norm)

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
        cols = df_focus.columns[df_focus.columns.str.contains(pat) & (~df_focus.columns.isin(["HIGH0", "LOW0"]))]
        df_focus[cols] = df_focus[cols].groupby(level="datetime").apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")]
        df_focus[cols] = df_focus[cols].apply(np.log).groupby(level="datetime").apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^RSQR")]
        df_focus[cols] = df_focus[cols].fillna(0).groupby(level="datetime").apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^MAX|^HIGH0")]
        df_focus[cols] = df_focus[cols].apply(lambda x: (x - 1) ** 0.5).groupby(level="datetime").apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^MIN|^LOW0")]
        df_focus[cols] = df_focus[cols].apply(lambda x: (1 - x) ** 0.5).groupby(level="datetime").apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^CORR|^CORD")]
        df_focus[cols] = df_focus[cols].apply(np.exp).groupby(level="datetime").apply(_feature_norm)

        cols = df_focus.columns[df_focus.columns.str.contains("^WVMA")]
        df_focus[cols] = df_focus[cols].apply(np.log1p).groupby(level="datetime").apply(_feature_norm)

        df[selected_cols] = df_focus.values

        TimeInspector.log_cost_time("Finished preprocessing data.")

        return df
