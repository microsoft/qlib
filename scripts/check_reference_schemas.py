# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Validate reference benchmark + RL configs against typed schemas.

Used by `make schema-check` and mirrored in engineering-elevation.yml.
Exits non-zero on any mismatch.
"""

from pathlib import Path

import yaml

from qlib.backtest.config_schema import validate_benchmark_workflow
from qlib.contrib.rl.config_schema import load_rl_backtest_config


def main() -> None:
    benchmark_path = Path("examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml")
    parsed = validate_benchmark_workflow(yaml.safe_load(benchmark_path.read_text(encoding="utf-8")))
    assert parsed.resolved_backtest.exchange_kwargs.deal_price == "$close", parsed.resolved_backtest
    print(f"benchmark schema OK: {benchmark_path}")

    rl_path = "examples/rl_order_execution/exp_configs/backtest_twap.yml"
    rl_config = load_rl_backtest_config(rl_path)
    assert rl_config.concurrency == 16, rl_config
    print(f"rl schema OK: {rl_path} ({rl_config.data_granularity})")


if __name__ == "__main__":
    main()
