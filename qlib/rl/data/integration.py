# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
TODO: This file is used to integrate NeuTrader with Qlib to run the existing projects.
TODO: The implementation here is kind of adhoc. It is better to design a more uniformed & general implementation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import cachetools
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.ops.high_freq import BFillNan, Cut, Date, DayCumsum, DayLast, FFillNan, IsInf, IsNull, Select
from qlib.data.dataset import DatasetH

dataset = None


class DataWrapper:
    def __init__(
        self,
        feature_dataset: DatasetH,
        backtest_dataset: DatasetH,
        columns_today: List[str],
        columns_yesterday: List[str],
        _internal: bool = False,
    ):
        assert _internal, "Init function of data wrapper is for internal use only."

        self.feature_dataset = feature_dataset
        self.backtest_dataset = backtest_dataset
        self.columns_today = columns_today
        self.columns_yesterday = columns_yesterday

    @cachetools.cached(  # type: ignore
        cache=cachetools.LRUCache(100),
        key=lambda _, stock_id, date, backtest: (stock_id, date.replace(hour=0, minute=0, second=0), backtest),
    )
    def get(self, stock_id: str, date: pd.Timestamp, backtest: bool = False) -> pd.DataFrame:
        start_time, end_time = date.replace(hour=0, minute=0, second=0), date.replace(hour=23, minute=59, second=59)
        dataset = self.backtest_dataset if backtest else self.feature_dataset
        return dataset.handler.fetch(pd.IndexSlice[stock_id, start_time:end_time], level=None)


def init_qlib(qlib_config: dict, part: str = None) -> None:
    """Initialize necessary resource to launch the workflow, including data direction, feature columns, etc..

    Parameters
    ----------
    qlib_config:
        Qlib configuration.

        Example::

            {
                "provider_uri_day": DATA_ROOT_DIR / "qlib_1d",
                "provider_uri_1min": DATA_ROOT_DIR / "qlib_1min",
                "feature_root_dir": DATA_ROOT_DIR / "qlib_handler_stock",
                "feature_columns_today": [
                    "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume",
                    "$bidV", "$bidV1", "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5",
                ],
                "feature_columns_yesterday": [
                    "$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1", "$bid_1", "$ask_1", "$volume_1",
                    "$bidV_1", "$bidV1_1", "$bidV3_1", "$bidV5_1", "$askV_1", "$askV1_1", "$askV3_1", "$askV5_1",
                ],
            }
    part
        Identifying which part (stock / date) to load.
    """

    global dataset  # pylint: disable=W0603

    def _convert_to_path(path: str | Path) -> Path:
        return path if isinstance(path, Path) else Path(path)

    provider_uri_map = {}
    if "provider_uri_day" in qlib_config:
        provider_uri_map["day"] = _convert_to_path(qlib_config["provider_uri_day"]).as_posix()
    if "provider_uri_1min" in qlib_config:
        provider_uri_map["1min"] = _convert_to_path(qlib_config["provider_uri_1min"]).as_posix()

    qlib.init(
        region=REG_CN,
        auto_mount=False,
        custom_ops=[DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut, DayCumsum],
        expression_cache=None,
        calendar_provider={
            "class": "LocalCalendarProvider",
            "module_path": "qlib.data.data",
            "kwargs": {
                "backend": {
                    "class": "FileCalendarStorage",
                    "module_path": "qlib.data.storage.file_storage",
                    "kwargs": {"provider_uri_map": provider_uri_map},
                },
            },
        },
        feature_provider={
            "class": "LocalFeatureProvider",
            "module_path": "qlib.data.data",
            "kwargs": {
                "backend": {
                    "class": "FileFeatureStorage",
                    "module_path": "qlib.data.storage.file_storage",
                    "kwargs": {"provider_uri_map": provider_uri_map},
                },
            },
        },
        provider_uri=provider_uri_map,
        kernels=1,
        redis_port=-1,
        clear_mem_cache=False,  # init_qlib will be called for multiple times. Keep the cache for improving performance
    )

    if part == "skip":
        return

    # this won't work if it's put outside in case of multiprocessing
    from qlib.data import D  # noqa pylint: disable=C0415,W0611

    if part is None:
        feature_path = Path(qlib_config["feature_root_dir"]) / "feature.pkl"
        backtest_path = Path(qlib_config["feature_root_dir"]) / "backtest.pkl"
    else:
        feature_path = Path(qlib_config["feature_root_dir"]) / "feature" / (part + ".pkl")
        backtest_path = Path(qlib_config["feature_root_dir"]) / "backtest" / (part + ".pkl")

    with feature_path.open("rb") as f:
        feature_dataset = pickle.load(f)
    with backtest_path.open("rb") as f:
        backtest_dataset = pickle.load(f)

    dataset = DataWrapper(
        feature_dataset,
        backtest_dataset,
        qlib_config["feature_columns_today"],
        qlib_config["feature_columns_yesterday"],
        _internal=True,
    )


def fetch_features(stock_id: str, date: pd.Timestamp, yesterday: bool = False, backtest: bool = False) -> pd.DataFrame:
    assert dataset is not None, "You must call init_qlib() before doing this."

    if backtest:
        fields = ["$close", "$volume"]
    else:
        fields = dataset.columns_yesterday if yesterday else dataset.columns_today

    data = dataset.get(stock_id, date, backtest)
    if data is None or len(data) == 0:
        # create a fake index, but RL doesn't care about index
        data = pd.DataFrame(0.0, index=np.arange(240), columns=fields, dtype=np.float32)  # FIXME: hardcode here
    else:
        data = data.rename(columns={c: c.rstrip("0") for c in data.columns})
        data = data[fields]
    return data
