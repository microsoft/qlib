# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qlib is an AI-oriented quantitative investment platform by Microsoft that supports machine learning modeling paradigms including supervised learning, market dynamics modeling, and reinforcement learning for financial data analysis and trading strategies.

## Development Commands

### Installation and Setup
```bash
# Install dependencies (requires numpy and cython first)
make prerequisite
make dependencies

# Development installation with all extras
make dev

# Install specific components
make lint      # Code quality tools
make rl        # Reinforcement learning dependencies  
make test      # Test dependencies
make analysis  # Analysis tools
make docs      # Documentation tools
```

### Code Quality
```bash
# Run all linting
make lint

# Individual linters
make black     # Code formatting
make pylint    # Static analysis
make flake8    # Style checking
make mypy      # Type checking

# Pre-commit setup
pip install -e .[dev]
pre-commit install
```

### Testing
```bash
# Run tests
pytest

# Specific test areas
python -m pytest tests/
python -m pytest tests/rl/
```

### Build and Package
```bash
make build     # Build wheel package
make upload    # Upload to PyPI
make clean     # Clean build artifacts
```

## Project Architecture

### Core Structure
- `qlib/` - Main package with modular components:
  - `data/` - Data processing, storage, and handlers
  - `model/` - ML models and ensemble methods
  - `backtest/` - Backtesting framework
  - `strategy/` - Trading strategies
  - `workflow/` - Experiment management
  - `rl/` - Reinforcement learning components
  - `contrib/` - Community contributions and extensions

### Key Concepts
- **Data Handlers**: Process financial data (Alpha158, Alpha360 datasets)
- **Models**: ML forecasting models (LightGBM, neural networks, etc.)
- **Strategies**: Trading logic (TopkDropout, signal-based)
- **Workflow**: End-to-end research pipeline using YAML configs
- **Executors**: Order execution simulation

### Configuration System
- Uses YAML workflow configs (see `examples/benchmarks/*/workflow_config_*.yaml`)
- Configuration handled by `qlib.config.Config` class
- Settings managed through `QSettings` with environment variable support (`QLIB_*`)

### Running Experiments
```bash
# Quick start with qrun tool
cd examples
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

# Custom workflows
python examples/workflow_by_code.py
python examples/run_all_model.py run --models=lightgbm
```

### Data Management
- Default data location: `~/.qlib/qlib_data/cn_data`
- Data download: `python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn`
- Health checking: `python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data`

### Extension Points
- Custom models in `qlib/contrib/model/`
- Custom strategies in `qlib/contrib/strategy/`
- Custom data handlers in `qlib/contrib/data/handler.py`
- Workflow templates in `examples/benchmarks/`

## Development Guidelines

### Code Standards
- Use Numpydoc style for docstrings
- Line length limit: 120 characters (enforced by black)
- Follow existing patterns in contrib modules
- Check available models/strategies before creating new ones

### Common Development Tasks
- Model development: Extend base classes in `qlib.model`
- Strategy development: Inherit from `BaseStrategy` 
- Data processing: Implement custom handlers extending `DataHandler`
- Testing: Add tests in `tests/` following existing patterns

### Pre-commit Hooks
The project uses pre-commit hooks for code formatting (black, flake8). Install with:
```bash
pip install -e .[dev]
pre-commit install
```