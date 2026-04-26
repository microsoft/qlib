from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import qlib
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.constant import REG_CN
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data import D
from qlib.utils import init_instance_by_config


def build_dataset_config(train_start: str, train_end: str, valid_start: str, valid_end: str, test_start: str, test_end: str):
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "instruments": "all",
                    "start_time": train_start,
                    "end_time": test_end,
                    "fit_start_time": train_start,
                    "fit_end_time": train_end,
                },
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    }


def build_model_config():
    return {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "num_threads": 4,
        },
    }


def build_benchmark(benchmark: str | None, start_time: str, end_time: str):
    if benchmark:
        return benchmark
    calendar = pd.DatetimeIndex(D.calendar(start_time=start_time, end_time=end_time, freq="day"))
    return pd.Series(0.0, index=calendar)


def main():
    parser = argparse.ArgumentParser(description="Run a minimal backtest on RQAlpha-converted Qlib data.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--valid-start", default="2019-01-01")
    parser.add_argument("--valid-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--test-end", default="2020-12-31")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--n-drop", type=int, default=1)
    parser.add_argument("--benchmark", default="", help="Optional benchmark instrument such as SH000300.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    provider_uri = str(Path(args.provider_uri).expanduser().resolve())
    print(f"provider_uri={provider_uri}", flush=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN, expression_cache=None, dataset_cache=None)
    print("qlib init done", flush=True)

    dataset = init_instance_by_config(
        build_dataset_config(
            args.train_start,
            args.train_end,
            args.valid_start,
            args.valid_end,
            args.test_start,
            args.test_end,
        )
    )
    print("dataset init done", flush=True)

    model_config = build_model_config()
    model_config["kwargs"]["device"] = args.device
    model = init_instance_by_config(model_config)
    print("model init done", flush=True)
    model.fit(dataset)
    print("model fit done", flush=True)

    pred = model.predict(dataset, segment="test")
    print(f"prediction size={pred.shape}", flush=True)

    strategy = TopkDropoutStrategy(signal=pred, topk=args.topk, n_drop=args.n_drop)
    trade_executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    benchmark = build_benchmark(args.benchmark.strip() or None, args.test_start, args.test_end)
    if isinstance(benchmark, str):
        print(f"benchmark={benchmark}", flush=True)
    else:
        print("benchmark=zero_return_series", flush=True)
    print("backtest start", flush=True)
    portfolio_metric_dict, indicator_dict = backtest(
        start_time=args.test_start,
        end_time=args.test_end,
        strategy=strategy,
        executor=trade_executor,
        benchmark=benchmark,
        account=10000000,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    )
    print("backtest done", flush=True)
    report_df = portfolio_metric_dict["1day"][0]
    print(report_df.tail(), flush=True)
    print(f"indicator keys={list(indicator_dict.keys())}", flush=True)


if __name__ == "__main__":
    main()
