import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, NamedTuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from utilsd.config import Registry

from .base import MarketPrice, FlowDirection


class PrefetchData(NamedTuple):
    raw_today: pd.DataFrame
    processed_today: pd.DataFrame
    processed_yesterday: pd.DataFrame


class IntraDaySingleAssetOrder(NamedTuple):
    """
    In the current context, raw should be a DataFrame with `datetime` as index and
    (at least) `$vwap0`, `$volume0`, `$close0` as columns.
    `processed` should be a DataFrame of 240x6, which is the same as `processed_prev`.
    """

    date: pd.Timestamp
    stock_id: str
    start_time: int
    end_time: int
    target: float
    flow_dir: int  # 0 for sell, 1 for buy
    prefetch_data: Optional[PrefetchData]

    def get_price(self, type: Literal['deal', 'close'] = 'deal'):
        if type == 'deal':
            return self.raw['$price'].values
        elif type == 'close':
            return self.raw['$close0'].values

    def get_volume(self):
        return self.raw['$volume0'].values

    def get_processed_data(self, type: Literal['today', 'yesterday'] = 'today'):
        if type == 'today':
            return self.processed.to_numpy()
        elif type == 'yesterday':
            return self.processed_prev.to_numpy()


@dataclass
class NumpyBasedIntraDaySingleAssetDataSchema(IntraDaySingleAssetDataSchema):
    """
    Mainly for acceleration purposes.
    """

    raw: np.ndarray
    processed: np.ndarray
    processed_prev: np.ndarray
    raw_name2idx: Dict[str, int]

    def get_price(self, type: Literal['deal', 'close'] = 'deal'):
        if type == 'deal':
            return self.raw[:, self.raw_name2idx['$price']]
        elif type == 'close':
            return self.raw[:, self.raw_name2idx['$close0']]

    def get_volume(self):
        return self.raw[:, self.raw_name2idx['$volume0']]

    def get_processed_data(self, type: Literal['today', 'yesterday'] = 'today'):
        if type == 'today':
            return self.processed
        elif type == 'yesterday':
            return self.processed_prev


def _infer_processed_data_column_names(shape):
    if shape == 16:
        return [
            "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume",
            "$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5"
        ]
    if shape == 6:
        return ["$high", "$low", "$open", "$close", "$vwap", "$volume"]
    elif shape == 5:
        return ["$high", "$low", "$open", "$close", "$volume"]
    raise ValueError(f"Unrecognized data shape: {shape}")


@DATASETS.register_module("intradaysa")
class IntraDaySingleAssetDataset(Dataset):
    def __init__(
        self,
        order_dir: Path,
        data_dir: Path,
        proc_data_dir: Path,
        total_time: int,
        proc_data_dim: int,
        default_start_time: Optional[int] = None,
        default_end_time: Optional[int] = None,
        deal_price_column: MarketPrice = MarketPrice.CLOSE,
        subset: Optional[str] = None,
        acceleration: bool = False
    ):
        if subset is not None:
            order_dir = order_dir / subset
        self.order_dir = order_dir
        self.data_dir = Path(data_dir)
        self.proc_data_dir = Path(proc_data_dir)
        self.total_time = total_time
        self.proc_data_dim = proc_data_dim
        self.orders = []
        self.deal_price_column = deal_price_column
        self.default_start_time = default_start_time
        self.default_end_time = default_end_time
        self.acceleration = acceleration
        if os.path.isfile(self.order_dir):
            self.orders = pd.read_pickle(self.order_dir)
        else:
            for file in self.order_dir.iterdir():
                order_data = pd.read_pickle(file)
                self.orders.append(order_data)
            self.orders = pd.concat(self.orders)

        if "start_time" not in self.orders:
            assert self.default_start_time is not None
            self.orders["start_time"] = self.default_start_time
        if "end_time" not in self.orders:
            assert self.default_end_time is not None
            self.orders["end_time"] = self.default_end_time

        # filter out orders with amount == 0
        self.orders = self.orders.query("amount > 0")

    def __len__(self):
        return len(self.orders)

    def _find_pickle(self, filename_without_suffix):
        suffix_list = [".pkl", ".pkl.backtest"]
        for suffix in suffix_list:
            path = filename_without_suffix.parent / (filename_without_suffix.name + suffix)
            if path.exists():
                return path
        raise FileNotFoundError(f'No file starting with "{filename_without_suffix}" found')

    def __getitem__(self, index):
        order = self.orders.iloc[index]
        date, stock_id = order.name
        target, start_time, end_time = order.amount, int(order.start_time), int(order.end_time)
        order_type = FlowDirection.LIQUIDATE if order.order_type == 0 else FlowDirection.ACQUIRE
        if self.deal_price_column in (MarketPrice.BID_OR_ASK, MarketPrice.BID_OR_ASK_FILL):
            deal_price_column_col = "$bid0" if order_type == FlowDirection.LIQUIDATE else "$ask0"
        elif self.deal_price_column == MarketPrice.CLOSE:
            deal_price_column_col = "$close0"
        backtest = pd.read_pickle(self._find_pickle(self.data_dir / stock_id))
        backtest = backtest.loc[pd.IndexSlice[stock_id, :, date]]
        backtest["$price"] = backtest[deal_price_column_col]

        if self.deal_price_column == MarketPrice.BID_OR_ASK_FILL:
            fill_price_col = "$bid0" if deal_price_column_col == "$ask0" else "$ask0"
            backtest["$price"] = backtest["$price"].replace(0, np.nan).fillna(backtest[fill_price_col])

        proc = pd.read_pickle(self._find_pickle(self.proc_data_dir / stock_id))
        cnames = _infer_processed_data_column_names(self.proc_data_dim)
        time, dim = self.total_time, self.proc_data_dim
        try:
            # new data format
            proc = proc.loc[pd.IndexSlice[stock_id, :, date]]
            assert len(proc) == time and len(proc.columns) == dim * 2
            proc_today = proc[cnames]
            proc_yesterday = proc[[f'{c}_1' for c in cnames]].rename(columns=lambda c: c[:-2])
        except (IndexError, KeyError):
            proc = proc.loc[pd.IndexSlice[stock_id, date]]
            assert time * dim * 2 == len(proc)
            proc_today = proc.to_numpy()[: time * dim].reshape((time, dim))
            proc_yesterday = proc.to_numpy()[time * dim:].reshape((time, dim))
            proc_today = pd.DataFrame(proc_today, index=backtest.index, columns=cnames)
            proc_yesterday = pd.DataFrame(proc_yesterday, index=backtest.index, columns=cnames)
        if self.acceleration:
            return NumpyBasedIntraDaySingleAssetDataSchema(
                date=date,
                stock_id=stock_id,
                target=target,
                flow_dir=order_type,
                raw=backtest.to_numpy(),
                processed=proc_today.to_numpy(),
                processed_prev=proc_yesterday.to_numpy(),
                start_time=start_time,
                end_time=end_time,
                raw_name2idx={c: i for i, c in enumerate(backtest.columns)}
            )
        else:
            return IntraDaySingleAssetDataSchema(
                date=date,
                stock_id=stock_id,
                target=target,
                flow_dir=order_type,
                raw=backtest,
                processed=proc_today,
                processed_prev=proc_yesterday,
                start_time=start_time,
                end_time=end_time,
            )


class InsDateSampler(Sampler[int]):
    def __init__(self, data_source: IntraDaySingleAssetDataset, shuffle: str = "none"):
        assert shuffle in ["none", "group", "all"]
        self.data_source = data_source
        self.shuffle = shuffle
        self._num_samples = None
        self.generator = None

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def _shuffle(self, lst, generator):
        index_list = torch.randperm(len(lst)).tolist()
        return [lst[i] for i in index_list]

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.shuffle == "none":
            yield from torch.arange(n).tolist()
        elif self.shuffle == "all":
            yield from torch.randperm(n, generator=generator).tolist()
        elif self.shuffle == "group":
            ins_list = self.data_source.orders.index.get_level_values("instrument")
            instruments = self._shuffle(sorted(set(ins_list)), generator)
            for ins in instruments:
                yield from np.where(ins_list == ins)[0].tolist()

    def __len__(self):
        return self.num_samples
