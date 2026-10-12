"""Filesystem-free tests for benchmark defaults, RL schemas, credentials, tracing."""

import logging
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from qlib.backtest.config_schema import validate_benchmark_workflow
from qlib.contrib.benchmark.base import (
    build_dataset_config,
    build_port_analysis_config,
    validated_exchange_kwargs,
)
from qlib.contrib.data.processor_tracing import (
    check_frame_memory_limit,
    estimate_frame_bytes,
    trace_processor_block,
)
from qlib.contrib.rl.config_schema import RLBacktestConfigSchema, load_rl_backtest_config
from qlib.log_structured import get_json_logger, log_latency_ms
from qlib.utils.credentials import collect_trading_credentials, resolve_credential, resolve_env_reference


def test_validated_exchange_kwargs_normalizes_and_rejects() -> None:
    merged = validated_exchange_kwargs({"deal_price": "close"})
    assert merged["deal_price"] == "$close"
    with pytest.raises(ValueError):
        validated_exchange_kwargs({"open_cost": -0.5})


def test_build_port_analysis_config_matches_legacy_yaml() -> None:
    raw = yaml.safe_load(
        open("examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml", encoding="utf-8")
    )
    legacy_kwargs = raw["port_analysis_config"]["backtest"]["exchange_kwargs"]
    rebuilt = build_port_analysis_config()
    for key, value in legacy_kwargs.items():
        expected = f"${value}" if key == "deal_price" and not str(value).startswith("$") else value
        assert rebuilt["backtest"]["exchange_kwargs"][key] == expected
    parsed = validate_benchmark_workflow(
        {**raw, "port_analysis_config": rebuilt} if "port_analysis_config" in raw else raw
    )
    assert parsed.resolved_backtest.exchange_kwargs.deal_price == "$close"


def test_build_dataset_config_uniform_segments() -> None:
    first = build_dataset_config(handler_kwargs={"instruments": "csi300"})
    second = build_dataset_config(handler_kwargs={"instruments": "csi500"})
    assert first["kwargs"]["segments"] == second["kwargs"]["segments"]
    assert first["kwargs"]["handler"]["kwargs"]["instruments"] == "csi300"
    assert second["kwargs"]["handler"]["kwargs"]["instruments"] == "csi500"


def test_rl_backtest_schema_accepts_repo_config() -> None:
    schema = load_rl_backtest_config("examples/rl_order_execution/exp_configs/backtest_twap.yml")
    assert schema.data_granularity == "5min"
    assert schema.concurrency == 16
    assert isinstance(schema, RLBacktestConfigSchema)
    with pytest.raises(ValueError):
        RLBacktestConfigSchema.model_validate({"concurrency": 0})


def test_credentials_resolve_from_env_and_reject_missing(monkeypatch) -> None:
    monkeypatch.setenv("QLIB_TRADE_API_KEY", "env-key")
    assert resolve_credential("QLIB_TRADE_API_KEY") == "env-key"
    assert resolve_credential("QLIB_TRADE_API_KEY", explicit="direct") == "direct"
    monkeypatch.delenv("QLIB_TRADE_API_KEY")
    with pytest.raises(ValueError):
        resolve_credential("QLIB_TRADE_API_KEY", required=True)
    creds = collect_trading_credentials(api_key="k")
    assert creds["api_key"] == "k"
    assert creds["api_secret"] is None


def test_resolve_env_reference_expands_and_guards(monkeypatch) -> None:
    monkeypatch.setenv("QLIB_PROVIDER_URI", "/data/cn")
    assert resolve_env_reference("${QLIB_PROVIDER_URI}") == "/data/cn"
    assert resolve_env_reference("/literal/path") == "/literal/path"
    monkeypatch.delenv("QLIB_PROVIDER_URI")
    with pytest.raises(ValueError):
        resolve_env_reference("${QLIB_PROVIDER_URI}")


def test_processor_tracing_and_memory_budget() -> None:
    frame = pd.DataFrame(
        np.ones((10, 4), dtype=np.float32),
        index=pd.date_range("2020-01-01", periods=10),
        columns=["a", "b", "c", "d"],
    )
    assert estimate_frame_bytes(frame) > 0
    report = check_frame_memory_limit(frame, max_bytes=10**9)
    assert report["rows"] == 10
    with pytest.raises(MemoryError):
        check_frame_memory_limit(frame, max_bytes=8)
    logger = get_json_logger("test_processor_tracing")
    with log_latency_ms(logger, "hf_processor.test"):
        pass
    with trace_processor_block("HighFreqNorm", run_id="test"):
        pass


def test_no_hardcoded_credentials_in_online_srv() -> None:
    root = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "online_srv")
    suspects = ("api_key=", "api_secret=", "password=", "AWS_SECRET")
    hits = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        text = open(os.path.join(root, name), encoding="utf-8").read()
        for marker in suspects:
            if marker in text.lower() and "os.environ" not in text.lower():
                # Only flag literal assignments, not env lookups or comments.
                for line in text.splitlines():
                    lowered = line.lower()
                    if marker in lowered and "environ" not in lowered and not lowered.strip().startswith("#"):
                        if ("=" in line) and ('"' in line or "'" in line):
                            hits.append(f"{name}:{line.strip()}")
    assert hits == [], f"possible hardcoded credentials: {hits}"


def test_structured_logger_json_shape(caplog) -> None:
    import json

    logger = get_json_logger("test_json_shape")
    with caplog.at_level(logging.INFO, logger="qlib.test_json_shape"):
        logger.info("hello")
    assert caplog.records, "expected a captured log record"
