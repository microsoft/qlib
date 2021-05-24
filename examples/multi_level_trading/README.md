# Multi-level Trading

This worflow is an example for multi-level trading.

## Introduction

Qlib supports backtesting of various strategies, including portfolio management strategies, order split strategies, model-based strategies (such as deep learning models), rule-based strategies, and RL-based strategies.

And, Qlib also supports multi-level trading and backtesting. It means that users can use different strategies to trade at different frequencies.

This example uses a DropoutTopkStrategy (a strategy based on the daily frequency Lightgbm model) in weekly frequency for portfolio generation. And, at the daily frequency level, this example uses SBBStrategyEMA (a rule-based strategy that uses EMA for decision-making) to split orders. 

## Usage

Start backtesting by running the following command:
```bash
    python workflow.py backtest
```

Start collecting data by running the following command:
```bash
    python workflow.py collect_data
```

