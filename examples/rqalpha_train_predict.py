from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import qlib
from qlib.constant import REG_CN
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
    parser = argparse.ArgumentParser(description="Train and predict with RQAlpha-converted Qlib data.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--valid-start", default="2019-01-01")
    parser.add_argument("--valid-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--test-end", default="2020-12-31")
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

    train_df = dataset.prepare("train")
    valid_df = dataset.prepare("valid")
    test_df = dataset.prepare("test")
    print(f"train shape={train_df.shape}", flush=True)
    print(f"valid shape={valid_df.shape}", flush=True)
    print(f"test shape={test_df.shape}", flush=True)

    model_config = build_model_config()
    model_config["kwargs"]["device"] = args.device
    model = init_instance_by_config(model_config)
    print("model init done", flush=True)
    model.fit(dataset)
    print("model fit done", flush=True)

    pred = model.predict(dataset, segment="test")
    print("prediction head:", flush=True)
    print(pred.head(), flush=True)


if __name__ == "__main__":
    main()
