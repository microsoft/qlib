# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Typed schemas for RL order-execution experiment configs (``exp_configs/*.yml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field


class QlibSectionSchema(BaseModel):
    model_config = {"extra": "allow"}

    provider_uri_5min: str = "./data/bin/"


class ExchangeSectionSchema(BaseModel):
    model_config = {"extra": "allow"}

    limit_threshold: Optional[float] = None
    deal_price: Union[str, List[str]] = "$close"
    volume_threshold: Optional[Any] = None


class StrategyEntrySchema(BaseModel):
    model_config = {"extra": "allow"}

    class_: str = Field(alias="class", default="TWAPStrategy")
    module_path: str = "qlib.contrib.strategy.rule_strategy"
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class RLBacktestConfigSchema(BaseModel):
    """Validated ``examples/rl_order_execution/exp_configs/*.yml`` mapping."""

    model_config = {"extra": "allow"}

    order_file: str
    start_time: str = "9:30"
    end_time: str = "14:54"
    data_granularity: str = "5min"
    qlib: QlibSectionSchema = Field(default_factory=QlibSectionSchema)
    exchange: ExchangeSectionSchema = Field(default_factory=ExchangeSectionSchema)
    strategies: Dict[str, StrategyEntrySchema] = Field(default_factory=dict)
    concurrency: int = Field(default=16, gt=0)
    output_dir: str = "outputs/twap/"


def load_rl_backtest_config(path: Union[str, Path]) -> RLBacktestConfigSchema:
    """Load and validate an RL order-execution YAML config from disk."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"RL config must be a mapping: {path}")
    return RLBacktestConfigSchema.model_validate(raw)


__all__ = [
    "ExchangeSectionSchema",
    "QlibSectionSchema",
    "RLBacktestConfigSchema",
    "StrategyEntrySchema",
    "load_rl_backtest_config",
]
