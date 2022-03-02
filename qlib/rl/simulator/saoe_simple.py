from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from .base import Simulator


class SingleOrderInitialState(NamedTuple):
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
    flow_dir: FlowDirection
    raw: pd.DataFrame
    processed: pd.DataFrame
    processed_prev: pd.DataFrame

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


class SingleAssetOrderExecution(Simulator):
    def __init__(self, initial: InitialStateType) -> None:
        pass

    def step(self, action: Any) -> None:
        raise NotImplementedError()

    def get_state(self) -> StateType:
        raise NotImplementedError()

    def done(self) -> bool:
        raise NotImplementedError()
