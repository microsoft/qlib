"""Evaluate the frozen Origin-1 checkpoint on a later rolling Valid segment.

This is temporal-transfer diagnosis only.  It neither trains nor selects a
checkpoint and never reads April or later observations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn import functional
from torch.utils.data import DataLoader, Subset

from examples.courage_strict_continuous_v1.evaluate_origin import _metrics
from qlib.contrib.data.courage_strict_v1.dataset import CourageStrictV1Dataset
from qlib.contrib.model.courage_strict_continuous_v1 import (
    atomic_json,
    load_config,
    sha256_file,
    validate_identity,
)
from qlib.contrib.model.courage_strict_v1 import _forward, _move
from qlib.contrib.model.patchtst_courage_strict_v1 import (
    HORIZONS_V1,
    PatchTSTCourageStrictV1,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/courage_strict_continuous_v1/config.json"
SOURCE_ORIGIN = "origin_2026_01"


class FrozenTransferEvaluationError(RuntimeError):
    """Raised when frozen transfer identity or population drifts."""


def _standardized_huber(
    frame: pd.DataFrame, scalers: dict[str, Any]
) -> tuple[dict[str, float], float]:
    values: dict[str, float] = {}
    for horizon in HORIZONS_V1:
        valid = frame[f"valid_{horizon}"].to_numpy(dtype=bool)
        target = frame.loc[valid, f"target_{horizon}"].to_numpy(dtype=np.float64)
        prediction = frame.loc[valid, f"prediction_{horizon}"].to_numpy(
            dtype=np.float64
        )
        stats = scalers["targets"][str(horizon)]
        standardized_target = (target - stats["median"]) / stats["iqr"]
        standardized_prediction = (prediction - stats["median"]) / stats["iqr"]
        value = functional.huber_loss(
            torch.from_numpy(standardized_prediction),
            torch.from_numpy(standardized_target),
            delta=1.0,
            reduction="mean",
        )
        values[str(horizon)] = float(value)
    return values, float(np.mean(list(values.values())))


def _render(target_origin: str, metrics: dict[str, Any]) -> str:
    summary = metrics["summary"]
    lines = [
        f"# Frozen Origin-1 → {target_origin} Temporal Transfer",
        "",
        "> 固定 Origin-1 step 2250 checkpoint 与 Origin-1 Train-only scaler；仅做后续滚动 Valid 推理，不训练、不选 checkpoint。",
        "",
        "## 汇总",
        "",
        f"- 标准化 Huber：`{summary['equal_head_standardized_huber']:.9f}`。",
        f"- raw RMSE：`{summary['equal_head_raw_rmse']:.8f}`；零预测 skill：`{summary['equal_head_zero_rmse_skill']:+.3%}`；Train 中位数 skill：`{summary['equal_head_train_median_rmse_skill']:+.3%}`。",
        f"- ACC：`{summary['equal_head_direction_accuracy']:.3%}`；多数类：`{summary['equal_head_majority_accuracy']:.3%}`；超额：`{summary['equal_head_accuracy_excess_vs_majority']:+.3%}`。",
        f"- BAcc：`{summary['equal_head_balanced_accuracy']:.3%}`；MCC：`{summary['equal_head_mcc']:+.5f}`；AUC：`{summary['equal_head_auc']:.3%}`。",
        f"- Pearson：`{summary['equal_head_pearson']:+.5f}`；Rank IC：`{summary['equal_head_rank_ic']:+.5f}`；Top−Bottom：`{summary['equal_head_top_bottom_spread']:+.6f}`。",
        "",
        "## 逐 horizon",
        "",
        "| H | Rows | RMSE skill | ACC | Majority | BAcc | MCC | AUC | Pearson | Rank IC | Spread |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS_V1:
        item = metrics["by_horizon"][str(horizon)]
        direction = item["direction"]
        lines.append(
            f"| {horizon} | {item['rows']:,} | {item['zero_rmse_skill']:+.3%} | "
            f"{direction['accuracy']:.3%} | {direction['majority_accuracy']:.3%} | "
            f"{direction['balanced_accuracy']:.3%} | {direction['mcc']:+.5f} | "
            f"{item['direction_auc']:.3%} | {item['pearson']:+.5f} | "
            f"{item['rank_ic_mean']:+.5f} | {item['top_bottom_spread_mean']:+.6f} |"
        )
    lines += [
        "",
        "这是滚动 Valid 的冻结迁移诊断，不是正式未见 Test，也不授权模型晋升。未读取 April、May、June 或以后数据。",
        "",
    ]
    return "\n".join(lines)


def evaluate(target_origin: str) -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world != 8 or torch.cuda.device_count() != 8:
        raise FrozenTransferEvaluationError("transfer requires exact 8-GPU torchrun")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    config = load_config(CONFIG)
    identity = validate_identity(config)
    if target_origin not in {"origin_2026_02", "origin_2026_03"}:
        raise FrozenTransferEvaluationError("target origin is outside bounded transfer")
    source_root = config.output_root / SOURCE_ORIGIN
    checkpoint_path = source_root / "best.pt"
    training_manifest = json.loads(
        (source_root / "_training_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("origin") != SOURCE_ORIGIN
        or checkpoint.get("global_step") != 2250
        or checkpoint.get("input_identity") != identity
        or checkpoint.get("june_or_later_read") is not False
        or sha256_file(checkpoint_path) != training_manifest.get("best_checkpoint_sha256")
    ):
        raise FrozenTransferEvaluationError("frozen checkpoint identity drift")
    segment = tuple(config.origins[target_origin]["valid"])
    if segment[1] > "2026-04-01":
        raise FrozenTransferEvaluationError("April-or-later transfer is forbidden")
    dataset = CourageStrictV1Dataset(
        provider_root=config.provider_root,
        segments={"valid": segment},
    )
    scaled = dataset.prepare("valid", scalers=checkpoint["scalers"], cache_capacity=8)
    row_indices = np.arange(rank, len(scaled), world, dtype=np.int64)
    loader = DataLoader(
        Subset(scaled, row_indices.tolist()),
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    model = PatchTSTCourageStrictV1(
        industry_vocab_size=int(checkpoint["industry_vocab_size"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    calendar_indices: list[np.ndarray] = []
    symbol_positions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            targets.append(batch["raw_targets"].numpy())
            masks.append(batch["target_mask"].numpy())
            calendar_indices.append(batch["calendar_index"].numpy())
            symbol_positions.append(batch["symbol_position"].numpy())
            moved = _move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = _forward(model, moved).float().cpu().numpy()
            for head, horizon in enumerate(HORIZONS_V1):
                stats = checkpoint["scalers"]["targets"][str(horizon)]
                prediction[:, head] = prediction[:, head] * stats["iqr"] + stats["median"]
            predictions.append(prediction)
    predicted = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks).astype(bool, copy=False)
    calendar_index = np.concatenate(calendar_indices)
    symbol_position = np.concatenate(symbol_positions)
    if not (
        len(predicted)
        == len(target)
        == len(mask)
        == len(calendar_index)
        == len(symbol_position)
        == len(row_indices)
    ):
        raise FrozenTransferEvaluationError("rank transfer population drift")
    output = source_root / "frozen_temporal_transfer" / target_origin
    if rank == 0:
        if output.exists():
            raise FrozenTransferEvaluationError(f"output already exists: {output}")
        (output / "fragments").mkdir(parents=True)
    dist.barrier()
    fragment = pd.DataFrame(
        {
            "row_index": row_indices,
            "calendar_index": calendar_index,
            "symbol_position": symbol_position,
        }
    )
    for head, horizon in enumerate(HORIZONS_V1):
        fragment[f"prediction_{horizon}"] = predicted[:, head]
        fragment[f"target_{horizon}"] = target[:, head]
        fragment[f"valid_{horizon}"] = mask[:, head]
    fragment_path = output / "fragments" / f"rank-{rank:02d}.parquet"
    fragment.to_parquet(fragment_path, index=False, compression="zstd")
    dist.barrier()
    if rank == 0:
        merged = pd.concat(
            [
                pd.read_parquet(output / "fragments" / f"rank-{item:02d}.parquet")
                for item in range(world)
            ],
            ignore_index=True,
        ).sort_values("row_index", kind="stable")
        if not np.array_equal(
            merged.row_index.to_numpy(dtype=np.int64),
            np.arange(len(scaled), dtype=np.int64),
        ):
            raise FrozenTransferEvaluationError("merged transfer row order drift")
        per_head_huber, equal_head_huber = _standardized_huber(
            merged, checkpoint["scalers"]
        )
        pseudo_best = {
            "global_step": int(checkpoint["global_step"]),
            "valid_equal_head_standardized_huber": equal_head_huber,
        }
        metrics = _metrics(
            merged,
            scalers=checkpoint["scalers"],
            best=pseudo_best,
            calendar=scaled.raw.provider.calendar,
        )
        metrics.update(
            {
                "schema_version": "courage_strict_continuous_frozen_transfer_metrics_v1",
                "role": "frozen_temporal_transfer_diagnostic",
                "is_unseen_test": False,
                "unseen_to_frozen_checkpoint": True,
                "used_for_checkpoint_selection": False,
                "source_origin": SOURCE_ORIGIN,
                "target_origin": target_origin,
                "target_segment": list(segment),
                "population_rows": len(merged),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "per_head_standardized_huber": per_head_huber,
                "read_before_exclusive": segment[1],
            }
        )
        merged.to_parquet(
            output / "predictions_and_targets.parquet",
            index=False,
            compression="zstd",
        )
        atomic_json(output / "metrics.json", metrics)
        report = _render(target_origin, metrics)
        (output / "TRANSFER_REPORT.md").write_text(report, encoding="utf-8")
        manifest = {
            "schema_version": "courage_strict_continuous_frozen_transfer_manifest_v1",
            "decision": "PASS_FROZEN_TEMPORAL_TRANSFER_DIAGNOSTIC",
            "source_origin": SOURCE_ORIGIN,
            "target_origin": target_origin,
            "target_segment": list(segment),
            "population_rows": len(merged),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics_sha256": sha256_file(output / "metrics.json"),
            "predictions_sha256": sha256_file(output / "predictions_and_targets.parquet"),
            "report_sha256": sha256_file(output / "TRANSFER_REPORT.md"),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "training_executed": False,
            "checkpoint_selection_executed": False,
            "april_or_later_read_executed": False,
            "refit_executed": False,
            "strategy_or_backtest_executed": False,
            "trading_executed": False,
            "remote_push_executed": False,
        }
        atomic_json(output / "_manifest.json", manifest)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-origin",
        required=True,
        choices=("origin_2026_02", "origin_2026_03"),
    )
    args = parser.parse_args()
    evaluate(args.target_origin)


if __name__ == "__main__":
    main()
