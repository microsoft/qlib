import os

import numpy as np
import pandas as pd
from qlib.data.dataset.processor import Processor
from qlib.data.dataset.utils import fetch_df_by_index
from typing import Dict


class HighFreqTrans(Processor):
    def __init__(self, dtype: str = "bool"):
        self.dtype = dtype

    def fit(self, df_features):
        pass

    def __call__(self, df_features):
        if self.dtype == "bool":
            return df_features.astype(np.int8)
        else:
            return df_features.astype(np.float32)


class HighFreqNorm(Processor):
    def __init__(
        self,
        fit_start_time: pd.Timestamp,
        fit_end_time: pd.Timestamp,
        feature_save_dir: str,
        norm_groups: Dict[str, int],
    ):

        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.feature_save_dir = feature_save_dir
        self.norm_groups = norm_groups

    def fit(self, df_features) -> None:
        if os.path.exists(self.feature_save_dir) and len(os.listdir(self.feature_save_dir)) != 0:
            return
        os.makedirs(self.feature_save_dir)
        fetch_df = fetch_df_by_index(df_features, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        del df_features
        index = 0
        names = {}
        for name, dim in self.norm_groups.items():
            names[name] = slice(index, index + dim)
            index += dim
        for name, name_val in names.items():
            df_values = fetch_df.iloc(axis=1)[name_val].values
            if name.endswith("volume"):
                df_values = np.log1p(df_values)
            self.feature_mean = np.nanmean(df_values)
            np.save(self.feature_save_dir + name + "_mean.npy", self.feature_mean)
            df_values = df_values - self.feature_mean
            self.feature_std = np.nanstd(np.absolute(df_values))
            np.save(self.feature_save_dir + name + "_std.npy", self.feature_std)
            df_values = df_values / self.feature_std
            np.save(self.feature_save_dir + name + "_vmax.npy", np.nanmax(df_values))
            np.save(self.feature_save_dir + name + "_vmin.npy", np.nanmin(df_values))
        return

    def __call__(self, df_features):
        if "date" in df_features:
            df_features.droplevel("date", inplace=True)
        df_values = df_features.values
        index = 0
        names = {}
        for name, dim in self.norm_groups.items():
            names[name] = slice(index, index + dim)
            index += dim
        for name, name_val in names.items():
            feature_mean = np.load(self.feature_save_dir + name + "_mean.npy")
            feature_std = np.load(self.feature_save_dir + name + "_std.npy")

            if name.endswith("volume"):
                df_values[:, name_val] = np.log1p(df_values[:, name_val])
            df_values[:, name_val] -= feature_mean
            df_values[:, name_val] /= feature_std
        df_features = pd.DataFrame(data=df_values, index=df_features.index, columns=df_features.columns)
        return df_features.fillna(0)
