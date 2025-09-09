# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qlib is an AI-oriented quantitative investment platform by Microsoft that supports diverse machine learning modeling paradigms for quantitative investment, including supervised learning, market dynamics modeling, and reinforcement learning. It provides a complete ML pipeline for data processing, model training, backtesting, and covers the entire quantitative investment chain.

## Architecture

### Core Modules
- **qlib/data/**: Data processing and storage layer, includes Cython extensions for high-performance operations
- **qlib/model/**: ML models and predictors for financial forecasting
- **qlib/workflow/**: High-level workflow orchestration and experiment management
- **qlib/backtest/**: Backtesting framework for strategy evaluation
- **qlib/strategy/**: Trading strategy implementations
- **qlib/rl/**: Reinforcement learning components for continuous investment decisions
- **qlib/contrib/**: Community-contributed models and workflows

### Key Dependencies
- Core: numpy, pandas (>=1.1), pyyaml, mlflow, lightgbm
- RL: torch, tianshou (<=0.4.10)
- Data: pyarrow, pymongo, redis
- Optimization: cvxpy, joblib

## Development Commands

### Setup
```bash
# Install prerequisites (Cython extensions)
make prerequisite

# Install package in editable mode
make install

# Install all development dependencies
make dev
```

### Testing
```bash
# Run tests with pytest
pytest tests/

# Run specific test file
pytest tests/test_workflow.py

# Exclude slow tests
pytest -m "not slow"
```

### Code Quality
```bash
# Run all linting checks
make lint

# Individual linters
make black      # Code formatting check (120 char limit)
make pylint     # Code quality check
make flake8     # Style guide enforcement
make mypy       # Type checking
make nbqa       # Jupyter notebook linting
```

### Building
```bash
# Build wheel package
make build

# Clean build artifacts
make clean

# Deep clean (removes virtual env, pre-commit hooks)
make deepclean
```

## Project-Specific Patterns

### Data Operations
- Custom Cython operations in `qlib/data/_libs/` for rolling/expanding window calculations
- Point-in-time (PIT) database support for avoiding look-ahead bias
- High-frequency data processing capabilities

### Model Development
- Models inherit from `qlib.model.base.Model`
- Use MLflow for experiment tracking
- Support for both tabular and time-series models

### Workflow Management
- YAML-based configuration in `qlib/workflow/`
- Recorder system for experiment tracking
- Task management with dependencies

### Testing Data
- Tests may require downloading financial data first
- Use fixture system in `tests/conftest.py`
- Mark slow tests with `@pytest.mark.slow`

## Important Notes

- Windows users: pywinpty is installed as a binary to avoid compilation issues
- Cython extensions (.pyx files) must be compiled before use
- MLflow tracking server may be needed for full workflow functionality
- Financial data needs to be downloaded separately for most examples