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

