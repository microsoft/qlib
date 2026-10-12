# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Typed Pydantic schemas for backtest and benchmark YAML configurations.

Keeps backward compatibility with existing ``workflow_config_*.yaml`` files:
all new fields have defaults and unknown legacy keys are ignored at the
top level, while the strictly-typed core (exchange/account costs) is validated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ExchangeKwargsSchema(BaseModel):
    """Validated ``exchange_kwargs`` section of a backtest config."""

    model_config = {"extra": "ignore"}

    limit_threshold: Optional[float] = None
    deal_price: str = "close"
    open_cost: float = Field(default=0.0005, ge=0.0, le=0.1)
    close_cost: float = Field(default=0.0015, ge=0.0, le=0.1)
    min_cost: float = Field(default=5.0, ge=0.0)
    impact_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    trade_unit: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("deal_price")
    @classmethod
    def _normalize_deal_price(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("deal_price must be a non-empty string")
        value = value.strip()
        return value if value.startswith("$") else "$" + value


class BacktestSectionSchema(BaseModel):
    """Validated ``backtest`` section shared by benchmark YAMLs."""

    model_config = {"extra": "allow"}

    start_time: str
    end_time: str
    account: float = Field(default=100_000_000, gt=0)
    benchmark: Any = "SH000300"
    exchange_kwargs: ExchangeKwargsSchema = Field(default_factory=ExchangeKwargsSchema)


class QlibInitSectionSchema(BaseModel):
    """Validated ``qlib_init`` header of benchmark YAMLs."""

    model_config = {"extra": "allow"}

    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: Literal["cn", "us", "tw"] = "cn"


class PortAnalysisSectionSchema(BaseModel):
    """Validated ``port_analysis_config`` wrapper holding strategy + backtest."""

    model_config = {"extra": "allow"}

    strategy: Dict[str, Any] = Field(default_factory=dict)
    backtest: BacktestSectionSchema


class BenchmarkWorkflowSchema(BaseModel):
    """Top-level schema for ``examples/benchmarks/*/workflow_config_*.yaml``.

    Legacy layout keeps ``backtest`` nested under ``port_analysis_config``;
    a top-level ``backtest`` alias is also accepted for forward compatibility.
    """

    model_config = {"extra": "allow"}

    qlib_init: QlibInitSectionSchema = Field(default_factory=QlibInitSectionSchema)
    market: Any = "csi300"
    benchmark: Any = "SH000300"
    backtest: Optional[BacktestSectionSchema] = None
    task: Dict[str, Any] = Field(default_factory=dict)
    data_handler_config: Dict[str, Any] = Field(default_factory=dict)
    port_analysis_config: Optional[PortAnalysisSectionSchema] = None

    @property
    def resolved_backtest(self) -> BacktestSectionSchema:
        """Return the effective backtest section from either layout."""
        if self.backtest is not None:
            return self.backtest
        if self.port_analysis_config is not None:
            return self.port_analysis_config.backtest
        raise ValueError("benchmark workflow config is missing a backtest section")


def validate_benchmark_workflow(config: Dict[str, Any]) -> BenchmarkWorkflowSchema:
    """Parse and validate a raw benchmark YAML mapping."""
    if not isinstance(config, dict):
        raise ValueError("benchmark workflow config must be a mapping")
    return BenchmarkWorkflowSchema.model_validate(config)


__all__: List[str] = [
    "BacktestSectionSchema",
    "BenchmarkWorkflowSchema",
    "ExchangeKwargsSchema",
    "PortAnalysisSectionSchema",
    "QlibInitSectionSchema",
    "validate_benchmark_workflow",
]
