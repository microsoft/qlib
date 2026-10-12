# ADR-0001: Backtest Execution / Data Pipeline Boundaries and Hardened Configs

Status: Accepted (Phase 1–6; see decisions 1–16)
Date: 2026-10-12

## Context

Qlib's backtest execution path (`qlib/backtest/executor.py`, `exchange.py`,
`account.py`, `position.py`) historically coupled order-matching and accounting
logic to the live data pipeline (`qlib.data.data.D`) and local
`~/.qlib/qlib_data` caches. That coupling made pure domain-logic unit testing
impossible without downloading market data, and benchmark YAMLs under
`examples/benchmarks/*` duplicated unvalidated `exchange_kwargs` maps.

The elevation brief requires: strict module boundaries, filesystem-free unit
tests for order matching/accounting, typed YAML validation, no hardcoded
secrets, and JSON observability for backtest runs.

## Decision

1. **Pure validation layer** — `qlib/backtest/validation.py` holds
   filesystem-free helpers (`validate_order`, `validate_position_consistency`,
   `sanitize_cache_path`, `check_deal_price_expression`). `Exchange` keeps
   orchestrating `D.features(...)`, but all structural invariants and cache-path
   traversal guards live in the pure layer and are covered by
   `tests/backtest/test_backtest_core.py` with zero dataset downloads.
2. **Typed benchmark schemas** — `qlib/backtest/config_schema.py` (Pydantic v2)
   models the *legacy* YAML shape exactly: `backtest` stays nested under
   `port_analysis_config`, with an optional top-level `backtest` alias exposed
   via `resolved_backtest`. No existing YAML is rewritten (backward compatible).
3. **Canonical benchmark defaults (Phase 2)** — `qlib/contrib/benchmark/base.py`
   centralizes `DEFAULT_EXCHANGE_KWARGS` / `DEFAULT_SEGMENTS` /
   `DEFAULT_BACKTEST` plus `build_port_analysis_config` and
   `build_dataset_config`, so new benchmarks inherit uniform dataset/handler
   sections instead of copy-pasting YAML blocks. Legacy YAMLs untouched.
4. **RL config schemas (Phase 2)** — `qlib/contrib/rl/config_schema.py`
   validates `examples/rl_order_execution/exp_configs/*.yml` (TWAP/PPO/OPDS):
   granularity, concurrency, exchange tuple-vs-string deal prices, strategy
   entries. Checked in the elevation workflow against `backtest_twap.yml`.
5. **Credential hygiene (Phase 2)** — `qlib/utils/credentials.py` resolves the
   `QLIB_TRADE_API_KEY` / `QLIB_TRADE_API_SECRET` / `QLIB_DATA_API_TOKEN` triple
   strictly from explicit args or environment (`${VAR}` expansion via
   `resolve_env_reference`); no hardcoded fallback exists. A repo test scans
   `examples/online_srv/*.py` for literal secret assignments.
6. **Central JSON logging** — `qlib/log_structured.py` (`JsonFormatter`,
   `get_json_logger`, `log_backtest_metrics`, `log_latency_ms`) is additive; the
   existing `qlib/log.py` `get_module_logger` path is untouched. Verbose
   `print()` call sites in `executor.py` remain for CLI UX but metrics lines
   should migrate to `log_backtest_metrics` (Sharpe / max-drawdown / latency).
7. **HF tracing + memory guards (Phase 2)** — `qlib/contrib/data/processor_tracing.py`
   adds `trace_processor_block` (JSON latency lines) and
   `estimate_frame_bytes` / `check_frame_memory_limit` budgets for large
   CSI300/500 frames. Existing `HighFreqNorm` behavior unchanged.
8. **Packaging/CI as additive gates** — `pyproject.toml` gains `[tool.ruff]`,
   `[tool.black]`, `[tool.mypy]` (strict only for the seven hardened modules),
   `[tool.pytest.ini_options]`, and `[tool.coverage]` (fail_under=80) sections.
   The `engineering-elevation.yml` workflow (py3.9–3.11) runs ruff +
   mypy-strict + filesystem-free tests (both phase files) + LightGBM/TWAP YAML
   schema validation. The existing `test_qlib_from_source*.yml` matrices are untouched.
9. **Container parity** — `Dockerfile` is multi-stage
   (`python:3.11-slim-bookworm`), non-root (`qlib` user), minimal runtime
   (`libgomp1` only). Secrets come from `--env-file .env` (see `.env.example`);
   nothing is baked into the image.
10. **Executor structured logging (Phase 3)** — `qlib/backtest/executor_logging.py`
    (`log_order_execution`) emits one JSON line per dealt order through
    `qlib.executor`; `SimulatorExecutor._collect_data` calls it on every order
    while keeping the legacy `verbose` print for CLI UX.
11. **Rolling-task budgets (Phase 3)** — `qlib/workflow/task/batch_limits.py`
    (`check_task_batch_limits`, `cap_task_batch`, `estimate_task_bytes`) caps
    `task_manager_rolling.py` dispatch at 512 tasks / 4 MiB per task by default.
12. **Handler-cache guards (Phase 3)** — `qlib/data/cache_guards.py`
    (`resolve_cache_file_uri`, `check_cache_file_budget`, `require_trusted_cache_load`)
    confines the `data_cache_demo.py` `file://` pickle round-trip to its base
    directory with a 2 GiB budget and explicit `trusted=True` opt-in.
13. **Execution/data decoupling (Phase 4)** — `qlib/backtest/data_provider.py`
    defines the `QuoteProvider` protocol with a lazy `QlibDataProvider` production
    adapter and an `InMemoryQuoteProvider` stub, so new execution-engine code can be
    unit-tested without `qlib.init()` or `~/.qlib` data. `Exchange` is untouched
    (backward compatible); adoption is forward-looking.
14. **Live seam wired (Phase 5)** — `Exchange.__init__` accepts optional
    `data_provider: QuoteProvider | None = None`. When `None` (default) every
    `D.instruments` / `D.features` call is byte-identical to before; when injected,
    instruments + quote frames come from the provider. Proven by
    `tests/backtest/test_exchange_with_provider.py`: a real `Exchange` deals a buy
    order end to end (price/cost/position assertions) and a NaN-`$close` bar deals
    zero — all with no dataset.
15. **Guard adoption (Phase 6)** — Phase 3 helpers now have production call sites:
    `task_manager_rolling.task_generating` caps/checks batches before TrainerRM
    fan-out (warn-and-cap at 512 tasks, raise only on 4 MiB oversized tasks);
    `data_cache_demo` validates the `file://` URI confinement and enforces the
    2 GiB cache budget after `to_pickle`, reusing one `cache_uri` in the
    generated task; `HighFreqNorm.fit` emits one JSON latency line via
    `trace_processor_block`. All guarded by source-regression tests in
    `tests/backtest/test_elevation_phase4.py`.
16. **YAML/pickle hardening (Phase 4)** — `qlib/contrib/rolling/ddgda.py` module
    constants and `examples/rl_order_execution/scripts/gen_pickle_data.py` moved from
    `yaml.FullLoader` to safe loaders (`yaml.safe_load` / ruamel `safe`); the
    untrusted-cache path already used `restricted_pickle_load`. `parse_config` in
    `qlib/utils/__init__.py` narrowed `except BaseException` → `except Exception`
    so `KeyboardInterrupt`/`SystemExit` propagate. Sweep confirmed zero bare
    `except:` clauses in `qlib/`.

## Consequences

- New execution-path code must be importable without `qlib.init()` / datasets;
  tests asserting that live in `tests/backtest/test_backtest_core.py` and
  `tests/backtest/test_elevation_phase2.py`.
- Benchmark YAML edits must keep passing `validate_benchmark_workflow` for the
  LightGBM Alpha158 reference config (checked in CI).
- Follow-ups (not in this change): migrate `executor.py:verbose print` to the
  JSON logger; add `pytest-cov` enforcement over all of `qlib/backtest/` once the
  environment provides full runtime deps; memory-profile
  `contrib/data/highfreq_processor.py` and `dataset.py` under CSI300/500 scale
  with real data budgets.
