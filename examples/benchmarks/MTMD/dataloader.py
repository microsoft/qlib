import torch
import numpy as np

class DataLoader:

    def __init__(self, df_feature, df_label, df_market_value, df_stock_index, batch_size=800, pin_memory=True, start_index = 0, device=None):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.df_market_value = df_market_value
        self.df_stock_index = df_stock_index
        self.device = device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            self.df_market_value = torch.tensor(self.df_market_value, dtype=torch.float, device=device)
            self.df_stock_index = torch.tensor(self.df_stock_index, dtype=torch.long, device=device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.start_index = start_index

        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):

        return len(self.daily_count)

    def iter_batch(self):
        if self.batch_size <= 0:
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i:i+self.batch_size] # NOTE: advanced indexing will cause copy

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        # for idx, count in zip(self.daily_index, self.daily_count):
        #     yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, slc):
        outs = self.df_feature[slc], self.df_label[slc][:,0], self.df_market_value[slc], self.df_stock_index[slc]
        # outs = self.df_feature[slc], self.df_label[slc]

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, device=self.device) for x in outs)

        return outs + (self.index[slc],)
