# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Run MA strategy backtest
"""
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.tests.data import GetData

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # load config
    import yaml
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), "workflow_config_ma_direct.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # run backtest
    with R.start(experiment_name="MA_Strategy"):
        # initialize model and dataset
        model = init_instance_by_config(config["task"]["model"])
        dataset = init_instance_by_config(config["task"]["dataset"])
        
        # train model
        model.fit(dataset)
        
        # initialize strategy and executor using get_strategy_executor
        from qlib.backtest import get_strategy_executor
        
        # prepare strategy config
        strategy_config = config["port_analysis_config"]["strategy"]
        # Replace <MODEL> and <DATASET> with actual instances
        strategy_config["kwargs"]["signal"] = (model, dataset)
        
        # prepare executor config
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True
            }
        }
        
        # get backtest parameters
        backtest_config = config["port_analysis_config"]["backtest"]
        start_time = backtest_config["start_time"]
        end_time = backtest_config["end_time"]
        benchmark = backtest_config["benchmark"]
        account = backtest_config["account"]
        exchange_kwargs = backtest_config["exchange_kwargs"]
        
        # initialize strategy and executor with common_infra
        strategy, executor = get_strategy_executor(
            start_time=start_time,
            end_time=end_time,
            strategy=strategy_config,
            executor=executor_config,
            benchmark=benchmark,
            account=account,
            exchange_kwargs=exchange_kwargs
        )
        
        # run backtest using backtest_loop
        from qlib.backtest.backtest import backtest_loop
        portfolio_dict, indicator_dict = backtest_loop(
            start_time=start_time,
            end_time=end_time,
            trade_strategy=strategy,
            trade_executor=executor
        )
        
        # record results
        print("Portfolio analysis results:")
        for key, value in portfolio_dict.items():
            print(f"{key}:")
            print(value[0])
        print("\nIndicator analysis:")
        for key, value in indicator_dict.items():
            print(f"{key}:")
            print(value[0])
