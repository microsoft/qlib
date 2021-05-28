# Multi-level Trading

This worflow is an example for multi-level trading.

## Introduction

Qlib supports backtesting of various strategies, including portfolio management strategies, order split strategies, model-based strategies (such as deep learning models), rule-based strategies, and RL-based strategies.

And, Qlib also supports multi-level trading and backtesting. It means that users can use different strategies to trade at different frequencies.


## Weekly Portfolio Generation and Daily Order Execution

This workflow provides an example that uses a DropoutTopkStrategy (a strategy based on the daily frequency Lightgbm model) in weekly frequency for portfolio generation and uses SBBStrategyEMA (a rule-based strategy that uses EMA for decision-making) to execute orders in daily frequency. 

### Usage

Start backtesting by running the following command:
```bash
    python workflow.py backtest
```

Start collecting data by running the following command:
```bash
    python workflow.py collect_data
```

## Daily Portfolio Generation and Minutely Order Execution

This workflow also provides a high-frequency example that uses a DropoutTopkStrategy for portfolio generation in daily frequency and uses SBBStrategyEMA to execute orders in minutely frequency. 

### Usage

Start backtesting by running the following command:
```bash
    python workflow.py backtest_highfreq
```