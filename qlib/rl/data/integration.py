# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
TODO: This file is used to integrate NeuTrader with Qlib to run the existing projects.
TODO: The implementation here is kind of adhoc. It is better to design a more uniformed & general implementation.
"""

from __future__ import annotations

from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.contrib.ops.high_freq import BFillNan, Cut, Date, DayCumsum, DayLast, FFillNan, IsInf, IsNull, Select


def init_qlib(qlib_config: dict) -> None:
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
    """

    def _convert_to_path(path: str | Path) -> Path:
        return path if isinstance(path, Path) else Path(path)

    provider_uri_map = {}
    for granularity in ["1min", "5min", "day"]:
        if f"provider_uri_{granularity}" in qlib_config:
            provider_uri_map[f"{granularity}"] = _convert_to_path(qlib_config[f"provider_uri_{granularity}"]).as_posix()

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
