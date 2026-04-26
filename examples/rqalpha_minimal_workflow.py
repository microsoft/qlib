from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import qlib
from qlib.backtest import backtest, executor
from qlib.constant import REG_CN
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
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


def main():
    parser = argparse.ArgumentParser(description="Run a minimal A-share workflow on RQAlpha-converted Qlib data.")
    parser.add_argument("--provider-uri", required=True, help="Qlib data directory converted from RQAlpha bundle.")
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--valid-start", default="2019-01-01")
    parser.add_argument("--valid-end", default="2020-12-31")
    parser.add_argument("--test-start", default="2021-01-01")
    parser.add_argument("--test-end", default="2021-12-31")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n-drop", type=int, default=2)
    args = parser.parse_args()

    provider_uri = str(Path(args.provider_uri).expanduser().resolve())
    qlib.init(provider_uri=provider_uri, region=REG_CN, expression_cache=None, dataset_cache=None)

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
    model = init_instance_by_config(build_model_config())

    train_df = dataset.prepare("train")
    valid_df = dataset.prepare("valid")
    test_df = dataset.prepare("test")
    print("train shape:", train_df.shape)
    print("valid shape:", valid_df.shape)
    print("test shape:", test_df.shape)

    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    print("prediction sample:")
    print(pred.head())

    strategy = TopkDropoutStrategy(signal=pred, topk=args.topk, n_drop=args.n_drop)
    report_df, positions = backtest(
        start_time=args.test_start,
        end_time=args.test_end,
        strategy=strategy,
        executor=executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True),
        benchmark=None,
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
    print("backtest report tail:")
    print(report_df.tail())
    print("positions sample dates:", list(positions.keys())[:3])


if __name__ == "__main__":
    main()
