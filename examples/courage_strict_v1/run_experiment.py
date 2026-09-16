"""Train Courage Strict V1 with Qlib Dataset, Model, provider and Recorder."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import torch

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.courage_strict_v1.dataset import CourageStrictV1Dataset
from qlib.contrib.data.courage_strict_v1.evaluate import (
    evaluate_predictions,
    render_markdown,
)
from qlib.contrib.model.courage_strict_v1 import CourageStrictV1Model
from qlib.workflow import R


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qlib-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.qlib_root.resolve()
    provider = root / "data/courage_strict_v1/qlib_provider"
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "formal V1 training requires CUDA; use --allow-cpu only for tests"
        )
    qlib.init(provider_uri={"1min": str(provider)}, region=REG_CN)
    dataset = CourageStrictV1Dataset(
        provider_root=provider,
        segments={
            "train": ("2025-07-01", "2026-03-02"),
            "valid": ("2026-03-02", "2026-04-01"),
        },
    )
    output = root / "artifacts/courage_strict_v1/training"
    model = CourageStrictV1Model(
        batch_size=args.batch_size,
        num_workers=args.workers,
        output_dir=str(output),
    )
    rank = int(os.environ.get("RANK", "0"))
    records = root / "artifacts/courage_strict_v1/qlib_records"
    records.mkdir(parents=True, exist_ok=True)
    artifact_root = records / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = artifact_root.as_uri()
    uri = f"sqlite:///{records / 'mlflow.db'}"
    context = (
        R.start(experiment_name="courage_strict_v1", uri=uri)
        if rank == 0
        else contextlib.nullcontext()
    )
    with context:
        if rank == 0:
            R.log_params(
                run_id="courage_strict_v1",
                dataset="Qlib_1min_bins",
                model="shared_PatchTST_7_heads",
                checkpoint_metric="valid_equal_head_standardized_Huber",
                batch_size=args.batch_size,
                workers=args.workers,
            )
        model.fit(dataset)
        if rank == 0:
            predictions = model.predict(dataset, segment="valid")
            raw = dataset.prepare("valid", cache_capacity=8)
            metrics = evaluate_predictions(raw=raw, predictions=predictions)
            output.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(
                output / "valid_predictions.parquet", compression="zstd"
            )
            (output / "valid_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            (output / "VALID_EVALUATION_REPORT.md").write_text(
                render_markdown(metrics), encoding="utf-8"
            )
            best = min(
                float(item["valid_equal_head_standardized_huber"])
                for item in model.history
            )
            R.log_metrics(valid_equal_head_standardized_huber=best)
            R.save_objects(
                **{
                    "model.pkl": model,
                    "valid_predictions.pkl": predictions,
                    "valid_metrics.json": metrics,
                }
            )


if __name__ == "__main__":
    main()
