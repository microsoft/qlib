import torch
import numpy as np
import pandas as pd

torch.set_default_dtype(torch.float32)


def get_daily_code(df):
    return df.reset_index(level=0).index.values


class DataLoader:
    def __init__(
        self,
        df_feature,
        df_label,
        region="CN",
        suffix="Train",
        batch_size=256,
        shuffle=False,
        device=None,
        risk_root="./dataset/riskdata",
    ):
        assert len(df_feature) == len(df_label)
        self.device = device

        self.df_feature = df_feature
        self.df_label = df_label

        self.feature = torch.from_numpy(self.df_feature.values).to(self.device).to(torch.float32)
        self.label = torch.from_numpy(self.df_label.values).to(self.device).to(torch.float32)

        self.region = region
        self.suffix = suffix
        self.root = risk_root
        self.index = df_label.index.to_series()
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.daily_cnt = self.index.groupby("datetime").size()
        self.daily_index = self.index.groupby("datetime").apply(get_daily_code)
        self.daily_date = self.daily_index.index.values
        self.end_idx = self.daily_cnt.cumsum().astype(int)
        self.start_idx = self.end_idx.shift(1).fillna(0).astype(int)

    @property
    def batch_length(self):
        return len(self.label) // self.batch_size

    @property
    def daily_length(self):
        return len(self.daily_cnt)

    def iter_batch(self):
        indices = np.arange(len(self.label))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i : i + self.batch_size]

    def iter_daily(self):
        indices = np.arange(len(self.daily_cnt))
        for i in indices:
            yield i, slice(self.start_idx[i], self.end_idx[i])

    def get_daily(self, i, slc):
        outs = self.feature[slc], self.label[slc][:, 0], self.get_variance(self.daily_date[i]), self.daily_index[i]
        return outs + (self.index[slc],)

    def get_batch(self, i, slc):
        outs = self.feature[slc], self.label[slc][:, 0], torch.cov(self.feature[slc]), -1  # No date in iter batch
        return outs + (self.index[slc],)

    def get_variance(self, date):
        date = pd.Timestamp(date)
        path = self.root + self.region + self.suffix + "/" + date.strftime("%Y%m%d") + "/factor_exp.pkl"
        df_variance = pd.read_pickle(path)
        return torch.from_numpy(df_variance.values).to(self.device).to(torch.float32)

    def get_daily_date(self, i):
        return pd.Timestamp(self.daily_date[i])
