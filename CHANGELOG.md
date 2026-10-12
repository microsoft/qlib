# Changelog

## Unreleased

### Engineering elevation (additive, backward-compatible)

- Packaging: `[tool.ruff]`, `[tool.black]`, `[tool.mypy]` (strict scoped to ten
  hardened modules), `[tool.pytest.ini_options]`, `[tool.coverage]` (`fail_under=80`)
  in `pyproject.toml`. No dependency bounds changed.
- New CI gate `.github/workflows/engineering-elevation.yml` (py3.9–3.11): ruff,
  `mypy --strict`, filesystem-free backtest tests with coverage, LightGBM Alpha158
  + TWAP RL schema validation. Existing source/pip matrices untouched.
- Backtest boundaries: `qlib/backtest/validation.py` (pure order/position/cache-path
  guards), `qlib/backtest/config_schema.py` (Pydantic benchmark schemas preserving the
  legacy `port_analysis_config.backtest` layout), `qlib/backtest/executor_logging.py`
  (JSON per-order lines dual-emitted alongside the legacy `verbose` print in
  `SimulatorExecutor._collect_data`).
- Benchmarks: `qlib/contrib/benchmark/base.py` canonical defaults/builders so new
  benchmarks share uniform dataset/handler sections; legacy YAMLs unchanged.
- RL configs: `qlib/contrib/rl/config_schema.py` validates
  `examples/rl_order_execution/exp_configs/*.yml`.
- Security: `qlib/utils/credentials.py` (env-or-explicit `QLIB_TRADE_API_KEY/SECRET`,
  `QLIB_DATA_API_TOKEN`; no hardcoded fallbacks) plus an `online_srv` secret-scan test;
  `qlib/data/cache_guards.py` (`file://` confinement, size budgets, `trusted=True`
  opt-in for handler pickles).
- Observability/perf: `qlib/log_structured.py` (JSON logs, Sharpe/drawdown/latency
  metrics), `qlib/contrib/data/processor_tracing.py` (HF block tracing + frame memory
  budgets), `qlib/workflow/task/batch_limits.py` (512-task / 4 MiB rolling-task caps).
- DX: `make elevation-gates|elevation-lint|elevation-types|elevation-tests|schema-check|env-check`,
  `.env.example`, multi-stage non-root `Dockerfile`, ruff `pre-commit` hooks,
  `docs/adr/0001-backtest-boundaries.md`, dev-guide + README onboarding sections.
- Security sweep: `yaml.FullLoader` → safe loaders in `contrib/rolling/ddgda.py` and
  `rl_order_execution/scripts/gen_pickle_data.py`; `except BaseException` → `except
  Exception` in `parse_config`; zero bare `except:` in `qlib/`; new `QuoteProvider`
  protocol (`qlib/backtest/data_provider.py`) for dataset-free execution tests.
- Live seam: `Exchange(data_provider=...)` injection with `None`-default legacy path;
  `tests/backtest/test_exchange_with_provider.py` deals real orders against an
  in-memory quote frame (buy + suspended-zero cases).
- Cross-platform builds: `scripts/build_wheels.py` (cibuildwheel, 6-OS/arch matrix,
  Linux aarch64 via QEMU), `scripts/validate_wheels.py` static wheel audit
  (glibc 2.28 floor, extension presence, cross-OS leakage), `build-wheels.yml`
  workflow (SHA-pinned, per-job TestPyPI publish), `make wheels|sdist` targets,
  and `docs/build/cross_platform_builds.rst` technical guide.
- Guard adoption: `task_manager_rolling` enforces 512-task/4 MiB batch budgets
  before fan-out; `data_cache_demo` validates `file://` URI confinement + 2 GiB
  budget; `HighFreqNorm.fit` emits JSON latency tracing — all source-regression
  tested in `test_elevation_phase4.py`.
- Offline conformance gate: `scripts/offline_check.py` (stdlib + PyYAML + tomli
  only, zero network) audits workflow SHA pins, Makefile targets, pyproject tool
  sections, bare-except/FullLoader sweeps, `.env` parity, and guide RST
  structure; `make offline-check`; 12 offline tests incl. a real-repo
  integration gate.

## Unreleased (upstream)

- **BREAKING:** New source builds restrict recorder artifact loading by default.
  Reloading executable artifacts requires verified source/storage and explicit
  `trusted=True` (CLI: `--trusted=True`). Supported data-only reads and fresh
  in-memory training need no opt-in. See the
  [artifact loading migration guide](https://qlib.readthedocs.io/en/latest/start/artifact_migration.html)
  for workflow, HIST and high-frequency cache upgrades.
- Merging into `main` affects source installs before a PyPI release. These changes
  remain unreleased until included in a tagged release; its versioned upgrade notes
  should link to the same guide.

### Configuration-driven execution

- Local `.py` imports require explicit boolean consent: top-level `trusted: true`
  on each component configuration, or `trusted=True` on a direct
  `get_module_by_module_path` call. Package imports and class objects are unchanged.
- Directory-root authorization from earlier PR revisions was removed. Consent is
  neither global nor inherited, and it does not authorize artifact loading.
- Feature expressions use a restricted AST interpreter; custom built-in TRA
  backbones and model-performance graph names use explicit mappings.

See the [configuration migration guide](docs/start/config_migration.rst) for the
upgrade checklist, Python/YAML examples, extension registration, and trusted
older file-model pickle recovery. These changes remain unreleased until included
in a tagged/PyPI release.

The full release history is maintained in [CHANGES.rst](CHANGES.rst).
