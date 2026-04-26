from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config


DEFAULT_SIMPLE_CACHE_PATH = Path("~/.cache/qlib_rqalpha_simple_cache").expanduser().resolve()
DEFAULT_MODEL_DIR = Path("artifacts/lgbm_models")


PRESETS = {
    "baseline": {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 100,
    },
    "conservative": {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "lambda_l1": 5.0,
        "lambda_l2": 5.0,
        "min_data_in_leaf": 200,
    },
    "wider": {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 100,
    },
}


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


def split_feature_label(df):
    feature = df["feature"].values
    label = df["label"].values
    if label.ndim == 2 and label.shape[1] == 1:
        label = np.squeeze(label)
    return feature, label


def parse_presets(value: str):
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in PRESETS]
    if unknown:
        raise ValueError(f"Unknown presets: {unknown}. Available presets: {sorted(PRESETS)}")
    return names


def add_qlib_runtime_args(parser: argparse.ArgumentParser, default_device: str = "gpu"):
    parser.add_argument("--device", choices=("cpu", "gpu"), default=default_device)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--dataset-cache",
        choices=("none", "simple"),
        default="simple",
        help="Use Qlib SimpleDatasetCache locally to avoid rebuilding the same feature dataset.",
    )
    parser.add_argument(
        "--local-cache-path",
        default=str(DEFAULT_SIMPLE_CACHE_PATH),
        help="Directory used by Qlib SimpleDatasetCache.",
    )
    parser.add_argument(
        "--clear-mem-cache",
        action="store_true",
        help="Clear in-process Qlib memory cache before init.",
    )


def build_qlib_init_kwargs(args, provider_uri: str):
    kwargs = {
        "provider_uri": provider_uri,
        "region": REG_CN,
        "expression_cache": None,
        "clear_mem_cache": args.clear_mem_cache,
    }
    if args.dataset_cache == "simple":
        kwargs["dataset_cache"] = "SimpleDatasetCache"
        kwargs["local_cache_path"] = str(Path(args.local_cache_path).expanduser().resolve())
    else:
        kwargs["dataset_cache"] = None
    return kwargs


def add_lgb_model_io_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory used to save or load LightGBM model artifacts.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained LightGBM models and metadata for future reuse.",
    )
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Load existing LightGBM models from --model-dir instead of retraining.",
    )


def build_lgb_model_prefix(args, preset_name: str) -> str:
    return "__".join(
        [
            preset_name,
            f"train_{args.train_start}_{args.train_end}",
            f"valid_{args.valid_start}_{args.valid_end}",
            f"test_{args.test_start}_{args.test_end}",
            f"inst_{args.instruments}",
            f"boost_{args.num_boost_round}",
            f"early_{args.early_stopping_rounds}",
            f"device_{args.device}",
        ]
    ).replace(":", "-")


def get_lgb_model_paths(args, preset_name: str):
    model_dir = Path(args.model_dir).expanduser().resolve()
    prefix = build_lgb_model_prefix(args, preset_name)
    return model_dir, model_dir / f"{prefix}.txt", model_dir / f"{prefix}.json"


def save_lgb_model_artifact(model: lgb.Booster, args, preset_name: str, extra_meta: dict | None = None):
    model_dir, model_path, meta_path = get_lgb_model_paths(args, preset_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    payload = {
        "preset": preset_name,
        "best_iteration": model.best_iteration,
    }
    if extra_meta:
        payload.update(extra_meta)
    meta_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return model_path, meta_path


def load_lgb_model_artifact(args, preset_name: str):
    _model_dir, model_path, meta_path = get_lgb_model_paths(args, preset_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, metadata, model_path, meta_path


def main():
    parser = argparse.ArgumentParser(description="Compare several LightGBM configs on RQAlpha-converted Qlib data.")
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--valid-start", default="2019-01-01")
    parser.add_argument("--valid-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--test-end", default="2020-12-31")
    add_qlib_runtime_args(parser, default_device="gpu")
    parser.add_argument("--instruments", default="all")
    parser.add_argument("--presets", default="baseline,conservative,wider")
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    add_lgb_model_io_args(parser)
    args = parser.parse_args()

    provider_uri = str(Path(args.provider_uri).expanduser().resolve())
    preset_names = parse_presets(args.presets)

    wall_start = time.perf_counter()
    print(f"provider_uri={provider_uri}", flush=True)
    print(f"presets={','.join(preset_names)}", flush=True)
    print(f"device={args.device}", flush=True)
    print(f"dataset_cache={args.dataset_cache}", flush=True)
    print(f"load_models={args.load_models}", flush=True)
    print(f"save_models={args.save_models}", flush=True)
    print(f"model_dir={Path(args.model_dir).expanduser().resolve()}", flush=True)
    if args.dataset_cache == "simple":
        print(f"local_cache_path={Path(args.local_cache_path).expanduser().resolve()}", flush=True)
    qlib.init(**build_qlib_init_kwargs(args, provider_uri))
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

    t1 = time.perf_counter()
    train_df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid_df = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    test_df = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    print(f"prepare splits done, elapsed={time.perf_counter() - t1:.2f}s", flush=True)
    print(f"train shape={train_df.shape}", flush=True)
    print(f"valid shape={valid_df.shape}", flush=True)
    print(f"test shape={test_df.shape}", flush=True)

    train_x, train_y = split_feature_label(train_df)
    valid_x, valid_y = split_feature_label(valid_df)
    test_x = test_df.values

    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    valid_set = lgb.Dataset(valid_x, label=valid_y, reference=train_set, free_raw_data=False)

    results = []
    for name in preset_names:
        fit_elapsed = 0.0
        pred_iteration = None
        if args.load_models:
            print(f"\n[{name}] load model start", flush=True)
            model, metadata, model_path, _meta_path = load_lgb_model_artifact(args, name)
            valid_l2 = float(metadata.get("valid_l2", float("nan")))
            train_l2 = float(metadata.get("train_l2", float("nan")))
            pred_iteration = int(metadata["best_iteration"]) if metadata.get("best_iteration") is not None else None
            print(
                f"[{name}] load model done: best_iteration={pred_iteration}, "
                f"valid_l2={valid_l2:.6f}, model_path={model_path}",
                flush=True,
            )
        else:
            params = {
                "objective": "mse",
                "metric": "l2",
                "verbosity": -1,
                "num_threads": args.num_threads,
                "device": args.device,
                **PRESETS[name],
            }
            evals_result = {}
            t_fit = time.perf_counter()
            print(f"\n[{name}] train start", flush=True)
            model = lgb.train(
                params,
                train_set,
                num_boost_round=args.num_boost_round,
                valid_sets=[train_set, valid_set],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(args.early_stopping_rounds),
                    lgb.log_evaluation(period=20),
                    lgb.record_evaluation(evals_result),
                ],
            )
            fit_elapsed = time.perf_counter() - t_fit
            valid_l2 = min(evals_result["valid"]["l2"])
            train_l2 = evals_result["train"]["l2"][model.best_iteration - 1]
            pred_iteration = model.best_iteration
            if args.save_models:
                model_path, _meta_path = save_lgb_model_artifact(
                    model,
                    args,
                    name,
                    extra_meta={"valid_l2": valid_l2, "train_l2": train_l2},
                )
                print(f"[{name}] model saved to {model_path}", flush=True)

        t_pred = time.perf_counter()
        pred = model.predict(test_x, num_iteration=pred_iteration)
        pred_elapsed = time.perf_counter() - t_pred
        results.append(
            {
                "preset": name,
                "best_iteration": pred_iteration if pred_iteration is not None else -1,
                "valid_l2": valid_l2,
                "train_l2": train_l2,
                "fit_elapsed": fit_elapsed,
                "predict_elapsed": pred_elapsed,
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred)),
            }
        )
        print(
            f"[{name}] done: best_iteration={pred_iteration}, "
            f"valid_l2={valid_l2:.6f}, fit_elapsed={fit_elapsed:.2f}s, "
            f"predict_elapsed={pred_elapsed:.2f}s",
            flush=True,
        )

    print("\nsummary:", flush=True)
    print("preset,best_iteration,valid_l2,train_l2,fit_elapsed,predict_elapsed,pred_mean,pred_std", flush=True)
    for row in sorted(results, key=lambda item: item["valid_l2"]):
        print(
            "{preset},{best_iteration},{valid_l2:.6f},{train_l2:.6f},{fit_elapsed:.2f},"
            "{predict_elapsed:.2f},{pred_mean:.6f},{pred_std:.6f}".format(**row),
            flush=True,
        )
    print(f"total elapsed={time.perf_counter() - wall_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
