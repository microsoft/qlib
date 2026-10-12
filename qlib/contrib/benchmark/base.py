# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Canonical benchmark workflow defaults shared by ``examples/benchmarks/*``.

Eliminates copy-pasted ``data_handler_config`` / ``exchange_kwargs`` blocks by
providing one typed builder. Existing YAML files are untouched (backward
compatible); new or generated benchmarks should build their task dicts from
here so dataset/handler instantiation stays uniform.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from qlib.backtest.config_schema import ExchangeKwargsSchema, validate_benchmark_workflow


DEFAULT_DATA_HANDLER_KWARGS: Dict[str, Any] = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi300",
}

DEFAULT_SEGMENTS: Dict[str, List[str]] = {
    "train": ["2008-01-01", "2014-12-31"],
    "valid": ["2015-01-01", "2016-12-31"],
    "test": ["2017-01-01", "2020-08-01"],
}

DEFAULT_EXCHANGE_KWARGS: Dict[str, Any] = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5.0,
}

DEFAULT_BACKTEST: Dict[str, Any] = {
    "start_time": "2017-01-01",
    "end_time": "2020-08-01",
    "account": 100_000_000,
    "benchmark": "SH000300",
    "exchange_kwargs": dict(DEFAULT_EXCHANGE_KWARGS),
}

DEFAULT_STRATEGY: Dict[str, Any] = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy",
    "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
}


def validated_exchange_kwargs(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return validated ``exchange_kwargs`` merged over canonical defaults."""
    merged = dict(DEFAULT_EXCHANGE_KWARGS)
    if overrides:
        merged.update(overrides)
    schema = ExchangeKwargsSchema.model_validate(merged)
    return schema.model_dump()


def build_port_analysis_config(
    backtest_overrides: Optional[Dict[str, Any]] = None,
    strategy_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a ``port_analysis_config`` (strategy + backtest) mapping."""
    backtest = deepcopy(DEFAULT_BACKTEST)
    if backtest_overrides:
        merged_exchange = dict(backtest["exchange_kwargs"])
        merged_exchange.update(backtest_overrides.pop("exchange_kwargs", {}))
        backtest.update(backtest_overrides)
        backtest["exchange_kwargs"] = merged_exchange
    backtest["exchange_kwargs"] = validated_exchange_kwargs(backtest["exchange_kwargs"])
    strategy = deepcopy(DEFAULT_STRATEGY)
    if strategy_overrides:
        strategy.update(strategy_overrides)
    return {"strategy": strategy, "backtest": backtest}


def build_dataset_config(
    handler_class: str = "Alpha158",
    handler_module: str = "qlib.contrib.data.handler",
    handler_kwargs: Optional[Dict[str, Any]] = None,
    segments: Optional[Dict[str, List[str]]] = None,
    dataset_class: str = "DatasetH",
    dataset_module: str = "qlib.data.dataset",
) -> Dict[str, Any]:
    """Build a uniform ``dataset`` task section for benchmark workflows."""
    merged_handler_kwargs = dict(DEFAULT_DATA_HANDLER_KWARGS)
    if handler_kwargs:
        merged_handler_kwargs.update(handler_kwargs)
    return {
        "class": dataset_class,
        "module_path": dataset_module,
        "kwargs": {
            "handler": {
                "class": handler_class,
                "module_path": handler_module,
                "kwargs": merged_handler_kwargs,
            },
            "segments": segments or dict(DEFAULT_SEGMENTS),
        },
    }


def validate_reference_benchmark(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a raw benchmark mapping and return its resolved backtest dump."""
    parsed = validate_benchmark_workflow(raw)
    return parsed.resolved_backtest.model_dump()


__all__ = [
    "DEFAULT_BACKTEST",
    "DEFAULT_DATA_HANDLER_KWARGS",
    "DEFAULT_EXCHANGE_KWARGS",
    "DEFAULT_SEGMENTS",
    "DEFAULT_STRATEGY",
    "build_dataset_config",
    "build_port_analysis_config",
    "validate_reference_benchmark",
    "validated_exchange_kwargs",
]
