# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import warnings
import numpy as np
import pandas as pd

from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH, DataHandler


device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)
    return x


def _create_ts_slices(index, seq_len):
    """
    create time series slices from pandas index

    Args:
        index (pd.MultiIndex): pandas multiindex with <instrument, datetime> order
        seq_len (int): sequence length
    """
    assert isinstance(index, pd.MultiIndex), "unsupported index type"
    assert seq_len > 0, "sequence length should be larger than 0"
    assert index.is_monotonic_increasing, "index should be sorted"

    # number of dates for each instrument
    sample_count_by_insts = index.to_series().groupby(level=0).size().values

    # start index for each instrument
    start_index_of_insts = np.roll(np.cumsum(sample_count_by_insts), 1)
    start_index_of_insts[0] = 0

    # all the [start, stop) indices of features
    # features between [start, stop) will be used to predict label at `stop - 1`
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_insts, sample_count_by_insts):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices, dtype="object")

    assert len(slices) == len(index)  # the i-th slice = index[i]

    return slices


def _get_date_parse_fn(target):
    """get date parse function

    This method is used to parse date arguments as target type.

    Example:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    if isinstance(target, int):
        _fn = lambda x: int(str(x).replace("-", "")[:8])  # 20200201
    elif isinstance(target, str) and len(target) == 8:
        _fn = lambda x: str(x).replace("-", "")[:8]  # '20200201'
    else:
        _fn = lambda x: x  # '2021-01-01'
    return _fn


def _maybe_padding(x, seq_len, zeros=None):
    """padding 2d <time * feature> data with zeros

    Args:
        x (np.ndarray): 2d data with shape <time * feature>
        seq_len (int): target sequence length
        zeros (np.ndarray): zeros with shape <seq_len * feature>
    """
    assert seq_len > 0, "sequence length should be larger than 0"
    if zeros is None:
        zeros = np.zeros((seq_len, x.shape[1]), dtype=np.float32)
    else:
        assert len(zeros) >= seq_len, "zeros matrix is not large enough for padding"
    if len(x) != seq_len:  # padding zeros
        x = np.concatenate([zeros[: seq_len - len(x), : x.shape[1]], x], axis=0)
    return x


class MTSDatasetH(DatasetH):
    """Memory Augmented Time Series Dataset

    Args:
        handler (DataHandler): data handler
        segments (dict): data split segments
        seq_len (int): time series sequence length
        horizon (int): label horizon
        num_states (int): how many memory states to be added
        memory_mode (str): memory mode (daily or sample)
        batch_size (int): batch size (<0 will use daily sampling)
        n_samples (int): number of samples in the same day
        shuffle (bool): whether shuffle data
        drop_last (bool): whether drop last batch < batch_size
        input_size (int): reshape flatten rows as this input_size (backward compatibility)
    """

    def __init__(
        self,
        handler,
        segments,
        seq_len=60,
        horizon=0,
        num_states=0,
        memory_mode="sample",
        batch_size=-1,
        n_samples=None,
        shuffle=True,
        drop_last=False,
        input_size=None,
        **kwargs,
    ):

        assert num_states == 0 or horizon > 0, "please specify `horizon` to avoid data leakage"
        assert memory_mode in ["sample", "daily"], "unsupported memory mode"
        assert memory_mode == "sample" or batch_size < 0, "daily memory requires daily sampling (`batch_size < 0`)"
        assert batch_size != 0, "invalid batch size"

        if batch_size > 0 and n_samples is not None:
            warnings.warn("`n_samples` can only be used for daily sampling (`batch_size < 0`)")

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_states = num_states
        self.memory_mode = memory_mode
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.input_size = input_size
        self.params = (batch_size, n_samples, drop_last, shuffle)  # for train/eval switch

        super().__init__(handler, segments, **kwargs)

    def setup_data(self, handler_kwargs: dict = None, **kwargs):

        super().setup_data(**kwargs)

        if handler_kwargs is not None:
            self.handler.setup_data(**handler_kwargs)

        # pre-fetch data and change index to <code, date>
        # NOTE: we will use inplace sort to reduce memory use
        try:
            df = self.handler._learn.copy()  # use copy otherwise recorder will fail
            # FIXME: currently we cannot support switching from `_learn` to `_infer` for inference
        except Exception:
            warnings.warn("cannot access `_learn`, will load raw data")
            df = self.handler._data.copy()
        df.index = df.index.swaplevel()
        df.sort_index(inplace=True)

        # convert to numpy
        self._data = df["feature"].values.astype("float32")
        np.nan_to_num(self._data, copy=False)  # NOTE: fillna in case users forget using the fillna processor
        self._label = df["label"].squeeze().values.astype("float32")
        self._index = df.index

        if self.input_size is not None and self.input_size != self._data.shape[1]:
            warnings.warn("the data has different shape from input_size and the data will be reshaped")
            assert self._data.shape[1] % self.input_size == 0, "data mismatch, please check `input_size`"

        # create batch slices
        self._batch_slices = _create_ts_slices(self._index, self.seq_len)

        # create daily slices
        daily_slices = {date: [] for date in sorted(self._index.unique(level=1))}  # sorted by date
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append(self._batch_slices[i])
        self._daily_slices = np.array(list(daily_slices.values()), dtype="object")
        self._daily_index = pd.Series(list(daily_slices.keys()))  # index is the original date index

        # add memory (sample wise and daily)
        if self.memory_mode == "sample":
            self._memory = np.zeros((len(self._data), self.num_states), dtype=np.float32)
        elif self.memory_mode == "daily":
            self._memory = np.zeros((len(self._daily_index), self.num_states), dtype=np.float32)
        else:
            raise ValueError(f"invalid memory_mode `{self.memory_mode}`")

        # padding tensor
        self._zeros = np.zeros((self.seq_len, max(self.num_states, self._data.shape[1])), dtype=np.float32)

    def _prepare_seg(self, slc, **kwargs):
        fn = _get_date_parse_fn(self._index[0][1])
        start_date = fn(slc.start)
        end_date = fn(slc.stop)
        obj = copy.copy(self)  # shallow copy
        # NOTE: Seriable will disable copy `self._data` so we manually assign them here
        obj._data = self._data  # reference (no copy)
        obj._label = self._label
        obj._index = self._index
        obj._memory = self._memory
        obj._zeros = self._zeros
        # update index for this batch
        date_index = self._index.get_level_values(1)
        obj._batch_slices = self._batch_slices[(date_index >= start_date) & (date_index <= end_date)]
        mask = (self._daily_index.values >= start_date) & (self._daily_index.values <= end_date)
        obj._daily_slices = self._daily_slices[mask]
        obj._daily_index = self._daily_index[mask]
        return obj

    def restore_index(self, index):
        return self._index[index]

    def restore_daily_index(self, daily_index):
        return pd.Index(self._daily_index.loc[daily_index])

    def assign_data(self, index, vals):
        if self.num_states == 0:
            raise ValueError("cannot assign data as `num_states==0`")
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        self._memory[index] = vals

    def clear_memory(self):
        if self.num_states == 0:
            raise ValueError("cannot clear memory as `num_states==0`")
        self._memory[:] = 0

    def train(self):
        """enable traning mode"""
        self.batch_size, self.n_samples, self.drop_last, self.shuffle = self.params

    def eval(self):
        """enable evaluation mode"""
        self.batch_size = -1
        self.n_samples = None
        self.drop_last = False
        self.shuffle = False

    def _get_slices(self):
        if self.batch_size < 0:  # daily sampling
            slices = self._daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:  # normal sampling
            slices = self._batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        indices = np.arange(len(slices))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(indices))[::batch_size]:
            if self.drop_last and i + batch_size > len(indices):
                break

            data = []  # store features
            label = []  # store labels
            index = []  # store index
            state = []  # store memory states
            daily_index = []  # store daily index
            daily_count = []  # store number of samples for each day

            for j in indices[i : i + batch_size]:

                # normal sampling: self.batch_size > 0 => slices is a list => slices_subset is a slice
                # daily sampling: self.batch_size < 0 => slices is a nested list => slices_subset is a list
                slices_subset = slices[j]

                # daily sampling
                # each slices_subset contains a list of slices for multiple stocks
                # NOTE: daily sampling is used in 1) eval mode, 2) train mode with self.batch_size < 0
                if self.batch_size < 0:

                    # store daily index
                    idx = self._daily_index.index[j]  # daily_index.index is the index of the original data
                    daily_index.append(idx)

                    # store daily memory if specified
                    # NOTE: daily memory always requires daily sampling (self.batch_size < 0)
                    if self.memory_mode == "daily":
                        slc = slice(max(idx - self.seq_len - self.horizon, 0), max(idx - self.horizon, 0))
                        state.append(_maybe_padding(self._memory[slc], self.seq_len, self._zeros))

                    # down-sample stocks and store count
                    if self.n_samples and 0 < self.n_samples < len(slices_subset):  # intraday subsample
                        slices_subset = np.random.choice(slices_subset, self.n_samples, replace=False)
                    daily_count.append(len(slices_subset))

                # normal sampling
                # each slices_subset is a single slice
                # NOTE: normal sampling is used in train mode with self.batch_size > 0
                else:
                    slices_subset = [slices_subset]

                for slc in slices_subset:

                    # legacy support for Alpha360 data by `input_size`
                    if self.input_size:
                        data.append(self._data[slc.stop - 1].reshape(self.input_size, -1).T)
                    else:
                        data.append(_maybe_padding(self._data[slc], self.seq_len, self._zeros))

                    if self.memory_mode == "sample":
                        state.append(_maybe_padding(self._memory[slc], self.seq_len, self._zeros)[: -self.horizon])

                    label.append(self._label[slc.stop - 1])
                    index.append(slc.stop - 1)

                    # end slices loop

                # end indices batch loop

            # concate
            data = _to_tensor(np.stack(data))
            state = _to_tensor(np.stack(state))
            label = _to_tensor(np.stack(label))
            index = np.array(index)
            daily_index = np.array(daily_index)
            daily_count = np.array(daily_count)

            # yield -> generator
            yield {
                "data": data,
                "label": label,
                "state": state,
                "index": index,
                "daily_index": daily_index,
                "daily_count": daily_count,
            }

        # end indice loop
