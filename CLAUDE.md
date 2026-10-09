# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Qlib (pyqlib) is an open-source, AI-oriented quantitative investment platform by Microsoft. It provides the full ML pipeline for quant research: data processing, factor engineering, model training, backtesting, and online serving. Supports Python 3.8–3.12 on Linux, macOS, and Windows.

## Build & Install

```bash
# Development install (editable + all extras)
make dev

# Or step-by-step:
make prerequisite   # compile Cython extensions (rolling, expanding)
make install        # pip install -e . (minimal)
make develop        # pip install -e .[dev] (adds pytest, statsmodels)
make lint           # install lint tools
make test           # install test deps

# Build wheel package
make build
```

The Cython extensions (`qlib/data/_libs/rolling.pyx`, `expanding.pyx`) compile C++ code for rolling/expanding window operations. They require `cython` and `numpy` headers.

## Test

```bash
# Run full test suite (excluding slow tests)
cd tests && python -m pytest . -m "not slow" --durations=0

# Run a single test file
cd tests && python -m pytest test_workflow.py -m "not slow"

# Run slow tests too
cd tests && python -m pytest .

# On macOS, thread limits are needed to prevent OpenMP segfaults:
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Tests are marked with `@pytest.mark.slow` for expensive tests. RL tests are only collected on Linux (`tests/conftest.py`). Tests require data downloaded to `~/.qlib/qlib_data/cn_data`.

## Lint

```bash
make black     # black -l 120 --check
make pylint    # pylint on qlib/ and scripts/ (many checks disabled, see Makefile)
make flake8    # flake8 on qlib/ (E501,F541,E266,E402,W503,E731,E203 ignored)
make mypy      # mypy on qlib/ (data, model, contrib, utils etc. excluded)
make nbqa      # Run black + pylint on notebooks
make lint      # all of the above
```

Line length is 120 chars. Pre-commit hooks run black + flake8 on push.

## Architecture

### Initialization

Everything starts with `qlib.init()` (configs in `qlib/config.py`). The global `C` object (`QlibConfig`) holds all configuration — provider URIs, cache settings, region parameters, logging, etc. Two modes exist: `client` (local data) and `server` (Redis-backed cache). Region-specific config (`REG_CN`, `REG_US`, `REG_TW`) controls trade units, limit thresholds, and deal prices.

Always call `qlib.init(provider_uri="...")` before any operation. The `qlib.auto_init()` helper finds `config.yaml` by walking up the directory tree.

### Data Layer (`qlib/data/`)

Providers abstract data access through a plugin architecture. The default is `LocalProvider` backed by `FileStorage` (binary `.bin` files keyed by instrument+field). Key abstractions:

- **`CalendarProvider`** — trading calendar
- **`InstrumentProvider`** — stock universe
- **`FeatureProvider`** — raw features
- **`ExpressionProvider`** — computed expressions (like a query engine for financial data)
- **`DatasetProvider`** — materialized datasets with caching
- **`PITProvider`** — point-in-time data (prevents lookahead bias)

The expression system (`qlib/data/ops.py`) defines operators like `Ref()`, `Mean()`, `Std()`, `Rsquare()`, etc. that compile into computations over financial time series. Custom operators can be registered at init time.

`qlib/data/dataset/` contains the data loading pipeline:
- `DataHandler` / `DataHandlerLP` — processes raw data into features/labels (train/valid/test segments)
- `DatasetH` — the standard PyTorch-style Dataset wrapper

### Model Layer (`qlib/model/`)

`Model` inherits from `BaseModel` and must implement `fit(dataset)` and `predict(dataset)`. Models are config-driven: each model class is instantiated from YAML config via `init_instance_by_config()`. The `qlib/model/trainer.py` `task_train()` function orchestrates training from config.

Model subdirectories:
- `qlib/model/ens/` — ensemble models
- `qlib/model/meta/` — meta-learning (DDG-DA, etc.)
- `qlib/model/interpret/` — interpretability tools
- `qlib/model/riskmodel/` — risk modeling

### Workflow & Experiment Management (`qlib/workflow/`)

Experiments track runs with MLflow as the backend. The global `R` (`QlibRecorder`) manages experiment lifecycle:

```python
with R.start(experiment_name="test", recorder_name="run1"):
    model.fit(dataset)
    R.log_metrics(mse=0.1, step=0)
    R.save_objects(model=model)
```

`qrun` CLI (`qlib/cli/run.py`) runs a full workflow from a YAML config file. It supports Jinja2 templating and base config inheritance via `BASE_CONFIG_PATH`.

### Backtesting (`qlib/backtest/`)

Components run in a nested decision framework:
- `exchange.py` — simulates trading with costs, limits, delays
- `executor.py` — executes decisions
- `decision.py` — trade decision representation
- `account.py` / `position.py` — portfolio tracking
- `signal.py` — signal generation from model predictions
- `report.py` / `profit_attribution.py` — performance analysis

### Strategy (`qlib/strategy/`)

Trading strategies transform model predictions into trade decisions. The base class is in `qlib/strategy/base.py`.

### Contrib (`qlib/contrib/`)

Community-contributed and research models, strategies, and workflows. Organized mirrors the core structure: `contrib/model/`, `contrib/strategy/`, `contrib/workflow/`, `contrib/data/`, `contrib/online/`, `contrib/rolling/`, `contrib/tuner/`. Production-grade models like LightGBM, GRU, LSTM, Transformer, HIST, etc. live here.

### Online Serving (`qlib/workflow/online/`)

Supports deploying trained models for live trading with automatic model rolling and updating.

## Config-Driven Design

Nearly everything is instantiated from YAML/dict config using `qlib.utils.init_instance_by_config()`. A config specifies `class`, `module_path`, and `kwargs`. This pattern is used for models, data handlers, strategies, and execution components.
