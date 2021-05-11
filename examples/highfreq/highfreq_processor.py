import numpy as np
import pandas as pd
from qlib.data.dataset.processor import Processor
from qlib.data.dataset.utils import fetch_df_by_index


class HighFreqNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time

    def fit(self, df_features):
        fetch_df = fetch_df_by_index(df_features, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        del df_features
        df_values = fetch_df.values
        names = {
            "price": slice(0, 10),
            "volume": slice(10, 12),
        }
        self.feature_med = {}
        self.feature_std = {}
        self.feature_vmax = {}
        self.feature_vmin = {}
        for name, name_val in names.items():
            part_values = df_values[:, name_val].astype(np.float32)
            if name == "volume":
                part_values = np.log1p(part_values)
            self.feature_med[name] = np.nanmedian(part_values)
            part_values = part_values - self.feature_med[name]
            self.feature_std[name] = np.nanmedian(np.absolute(part_values)) * 1.4826 + 1e-12
            part_values = part_values / self.feature_std[name]
            self.feature_vmax[name] = np.nanmax(part_values)
            self.feature_vmin[name] = np.nanmin(part_values)

    def __call__(self, df_features):
        df_features.set_index("date", append=True, drop=True, inplace=True)
        df_values = df_features.values
        names = {
            "price": slice(0, 10),
            "volume": slice(10, 12),
        }

        for name, name_val in names.items():
            if name == "volume":
                df_values[:, name_val] = np.log1p(df_values[:, name_val])
            df_values[:, name_val] -= self.feature_med[name]
            df_values[:, name_val] /= self.feature_std[name]
            slice0 = df_values[:, name_val] > 3.0
            slice1 = df_values[:, name_val] > 3.5
            slice2 = df_values[:, name_val] < -3.0
            slice3 = df_values[:, name_val] < -3.5

            df_values[:, name_val][slice0] = (
                3.0 + (df_values[:, name_val][slice0] - 3.0) / (self.feature_vmax[name] - 3) * 0.5
            )
            df_values[:, name_val][slice1] = 3.5
            df_values[:, name_val][slice2] = (
                -3.0 - (df_values[:, name_val][slice2] + 3.0) / (self.feature_vmin[name] + 3) * 0.5
            )
            df_values[:, name_val][slice3] = -3.5
        idx = df_features.index.droplevel("datetime").drop_duplicates()
        idx.set_names(["instrument", "datetime"], inplace=True)

        # Reshape is specifically for adapting to RL high-freq executor
        feat = df_values[:, [0, 1, 2, 3, 4, 10]].reshape(-1, 6 * 240)
        feat_1 = df_values[:, [5, 6, 7, 8, 9, 11]].reshape(-1, 6 * 240)
        df_new_features = pd.DataFrame(
            data=np.concatenate((feat, feat_1), axis=1),
            index=idx,
            columns=["FEATURE_%d" % i for i in range(12 * 240)],
        ).sort_index()
        return df_new_features
