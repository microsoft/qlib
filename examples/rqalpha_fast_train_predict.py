from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config


def build_dataset_config(
    instruments: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
):
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "instruments": instruments,
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


def build_model_config(device: str, num_threads: int):
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
            "num_threads": num_threads,
            "device": device,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Faster train and predict workflow for Qlib daily A-share research.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--valid-start", default="2019-01-01")
    parser.add_argument("--valid-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--test-end", default="2020-12-31")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--instruments", default="all")
    parser.add_argument("--dataset-cache", choices=("none", "simple"), default="none")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--show-shapes", action="store_true")
    args = parser.parse_args()

    provider_uri = str(Path(args.provider_uri).expanduser().resolve())
    cache_dir = args.cache_dir.strip() or str(Path(provider_uri).joinpath("_local_dataset_cache"))

    init_kwargs = {
        "provider_uri": provider_uri,
        "region": REG_CN,
        "expression_cache": None,
        "dataset_cache": None,
    }
    if args.dataset_cache == "simple":
        init_kwargs["dataset_cache"] = "SimpleDatasetCache"
        init_kwargs["local_cache_path"] = cache_dir

    wall_start = time.perf_counter()
    print(f"provider_uri={provider_uri}", flush=True)
    print(f"dataset_cache={args.dataset_cache}", flush=True)
    if args.dataset_cache == "simple":
        print(f"cache_dir={cache_dir}", flush=True)
    qlib.init(**init_kwargs)
    print("qlib init done", flush=True)

    t0 = time.perf_counter()
    dataset = init_instance_by_config(
        build_dataset_config(
            args.instruments,
            args.train_start,
            args.train_end,
            args.valid_start,
            args.valid_end,
            args.test_start,
            args.test_end,
        )
    )
    print(f"dataset init done, elapsed={time.perf_counter() - t0:.2f}s", flush=True)

    model = init_instance_by_config(build_model_config(args.device, args.num_threads))
    print("model init done", flush=True)

    t1 = time.perf_counter()
    model.fit(dataset)
    print(f"model fit done, elapsed={time.perf_counter() - t1:.2f}s", flush=True)

    t2 = time.perf_counter()
    pred = model.predict(dataset, segment="test")
    print(f"predict done, elapsed={time.perf_counter() - t2:.2f}s", flush=True)

    if args.show_shapes:
        train_df, valid_df, test_df = dataset.prepare(
            ["train", "valid", "test"], col_set=["feature", "label"], data_key="learn"
        )
        print(f"train shape={train_df.shape}", flush=True)
        print(f"valid shape={valid_df.shape}", flush=True)
        print(f"test shape={test_df.shape}", flush=True)

    print(f"prediction size={pred.shape}", flush=True)
    print("prediction head:", flush=True)
    print(pred.head(), flush=True)
    print(f"total elapsed={time.perf_counter() - wall_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
