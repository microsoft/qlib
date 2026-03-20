# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Simple implementation of MA strategy using QLib's built-in components
"""
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.tests.data import GetData
import pandas as pd

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # Create a simple MA strategy using built-in TopkDropoutStrategy
    # We'll use a custom data handler that calculates MA signals
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader

    # Create a custom data loader for MA signals
    class MADataLoader(QlibDataLoader):
        @staticmethod
        def get_feature_config():
            fields = [
                "Mean($close, 5)/$close",  # MA5
                "Mean($close, 20)/$close", # MA20
                "(Mean($close, 5) - Mean($close, 20))/$close", # MA_DIFF
                "If(Mean($close, 5) > Mean($close, 20), 1, -1)" # MA_SIGNAL
            ]
            names = ["MA5", "MA20", "MA_DIFF", "MA_SIGNAL"]
            return fields, names

    # Create a custom data handler
    class MAHandler(DataHandlerLP):
        def __init__(self, instruments="csi300", **kwargs):
            data_loader = {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": MADataLoader.get_feature_config(),
                        "label": ["Ref($close, -2)/Ref($close, -1) - 1"],
                    },
                    "freq": "day",
                },
            }
            super().__init__(
                instruments=instruments,
                data_loader=data_loader,
                **kwargs,
            )

    # Create dataset
    dataset = DatasetH(
        handler=MAHandler(
            instruments="csi300",
            start_time="2017-01-01",
            end_time="2020-08-01",
        ),
        segments={
            "train": ["2017-01-01", "2018-12-31"],
            "valid": ["2019-01-01", "2019-12-31"],
            "test": ["2020-01-01", "2020-08-01"]
        }
    )

    # Create a simple linear model
    from qlib.contrib.model.linear import LinearModel
    model = LinearModel(estimator="ols")

    # Train the model
    print("Training model...")
    model.fit(dataset)

    # Get predictions
    print("Getting predictions...")
    pred = model.predict(dataset)

    # Create backtest config
    backtest_config = {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5
        }
    }

    # Create strategy
    from qlib.contrib.strategy import TopkDropoutStrategy
    strategy = TopkDropoutStrategy(
        signal=(model, dataset),
        topk=100,
        n_drop=10,
        risk_degree=0.95
    )

    # Create executor
    from qlib.backtest.executor import SimulatorExecutor
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        strategy=strategy,
        backtest=backtest_config
    )

    # Run backtest
    print("Running backtest...")
    from qlib.backtest.backtest import backtest_loop
    portfolio_dict, indicator_dict = backtest_loop(
        start_time=backtest_config["start_time"],
        end_time=backtest_config["end_time"],
        trade_strategy=strategy,
        trade_executor=executor
    )

    # Print results
    print("Backtest completed!")
    print("\nPortfolio metrics:")
    for key, value in portfolio_dict.items():
        print(f"{key}:")
        print(value[0])
    print("\nIndicator analysis:")
    for key, value in indicator_dict.items():
        print(f"{key}:")
        print(value[0])
