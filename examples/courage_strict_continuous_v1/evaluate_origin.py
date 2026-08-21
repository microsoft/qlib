"""Evaluate one completed continuous-training origin on its full rolling Valid.

The rolling Valid population is a checkpoint-selection population, not an
unseen Test.  Target masks are taken directly from the dataset batches so the
per-horizon label-maturity purge is preserved exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

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


class ContinuousOriginEvaluationError(RuntimeError):
    """Raised when evaluation identity, population, or metrics drift."""


def _auc(target_up: np.ndarray, score: np.ndarray) -> float:
    positive = int(target_up.sum())
    negative = int(len(target_up) - positive)
    if positive == 0 or negative == 0:
        return math.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
    return float(
        (ranks[target_up].sum() - positive * (positive + 1) / 2) / (positive * negative)
    )


def _direction(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    active = target != 0.0
    truth = target[active] > 0
    pred_up = prediction[active] > 0
    pred_down = prediction[active] < 0
    tp = int((truth & pred_up).sum())
    fn = int((truth & ~pred_up).sum())
    tn = int((~truth & pred_down).sum())
    fp = int((~truth & ~pred_down).sum())
    count = int(active.sum())
    recall_up = tp / (tp + fn) if tp + fn else math.nan
    recall_down = tn / (tn + fp) if tn + fp else math.nan
    majority = max(float(truth.mean()), float((~truth).mean())) if count else math.nan
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    accuracy = (tp + tn) / count if count else math.nan
    return {
        "active_rows": count,
        "flat_target_rows": int((target == 0).sum()),
        "accuracy": accuracy,
        "majority_accuracy": majority,
        "accuracy_excess_vs_majority": accuracy - majority,
        "balanced_accuracy": (recall_up + recall_down) / 2,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "recall_up": recall_up,
        "recall_down": recall_down,
        "target_up_rate": float(truth.mean()) if count else math.nan,
        "prediction_up_rate": float(pred_up.mean()) if count else math.nan,
        "prediction_zero_rows": int((prediction[active] == 0).sum()),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def _cross_sectional(
    calendar_index: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "calendar_index": calendar_index,
            "target": target,
            "prediction": prediction,
        }
    )
    groups = frame.groupby("calendar_index", sort=False)
    rank_ic = (
        groups.apply(
            lambda value: value["prediction"].corr(value["target"], method="spearman"),
            include_groups=False,
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    frame["prediction_rank"] = groups["prediction"].rank(method="average", pct=True)
    top = (
        frame.loc[frame.prediction_rank >= 0.9]
        .groupby("calendar_index", sort=False)
        .target.mean()
    )
    bottom = (
        frame.loc[frame.prediction_rank <= 0.1]
        .groupby("calendar_index", sort=False)
        .target.mean()
    )
    spread = top.subtract(bottom, fill_value=np.nan).dropna()
    return {
        "rank_ic_mean": float(rank_ic.mean()),
        "rank_ic_median": float(rank_ic.median()),
        "rank_ic_std_ddof1": float(rank_ic.std(ddof=1)),
        "rank_ic_positive_timestamp_ratio": float((rank_ic > 0).mean()),
        "rank_ic_timestamps": len(rank_ic),
        "top_bottom_spread_mean": float(spread.mean()),
        "top_bottom_spread_positive_timestamp_ratio": float((spread > 0).mean()),
        "top_bottom_timestamps": len(spread),
    }


def _metrics(
    frame: pd.DataFrame,
    *,
    scalers: dict[str, Any],
    best: dict[str, Any],
    calendar: pd.DatetimeIndex,
) -> dict[str, Any]:
    loss_function = str(best.get("loss_function", "HUBER")).upper()
    selection_key = f"valid_equal_head_standardized_{loss_function.lower()}"
    if selection_key not in best:
        raise ContinuousOriginEvaluationError("best checkpoint loss metric drift")
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS_V1:
        valid = frame[f"valid_{horizon}"].to_numpy(dtype=bool)
        target = frame.loc[valid, f"target_{horizon}"].to_numpy(dtype=np.float64)
        prediction = frame.loc[valid, f"prediction_{horizon}"].to_numpy(
            dtype=np.float64
        )
        indices = frame.loc[valid, "calendar_index"].to_numpy(dtype=np.int64)
        if (
            not len(target)
            or not np.isfinite(target).all()
            or not np.isfinite(prediction).all()
        ):
            raise ContinuousOriginEvaluationError(
                f"invalid values for horizon {horizon}"
            )
        error = prediction - target
        zero_rmse = float(np.sqrt(np.square(target).mean()))
        train_constant = float(scalers["targets"][str(horizon)]["median"])
        train_constant_rmse = float(np.sqrt(np.square(target - train_constant).mean()))
        centered_target = target - target.mean()
        centered_prediction = prediction - prediction.mean()
        pearson_denominator = math.sqrt(
            float(
                np.square(centered_target).sum() * np.square(centered_prediction).sum()
            )
        )
        variance = float(np.square(centered_prediction).sum())
        daily = (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(calendar[indices]).date,
                    "squared_error": np.square(error),
                }
            )
            .groupby("date", sort=False)
            .squared_error.mean()
            .pow(0.5)
        )
        direction = _direction(target, prediction)
        active = target != 0.0
        item = {
            "rows": len(target),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()),
            "zero_rmse": zero_rmse,
            "zero_rmse_skill": 1.0
            - float(np.sqrt(np.square(error).mean())) / zero_rmse,
            "train_median_constant": train_constant,
            "train_median_rmse": train_constant_rmse,
            "train_median_rmse_skill": 1.0
            - float(np.sqrt(np.square(error).mean())) / train_constant_rmse,
            "pearson": (
                float(
                    (centered_target * centered_prediction).sum() / pearson_denominator
                )
                if pearson_denominator
                else 0.0
            ),
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std(ddof=0)),
            "target_mean": float(target.mean()),
            "target_std": float(target.std(ddof=0)),
            "calibration_slope_target_on_prediction": (
                float((centered_prediction * centered_target).sum() / variance)
                if variance
                else 0.0
            ),
            "daily_rmse_mean": float(daily.mean()),
            "daily_rmse_std_ddof1": float(daily.std(ddof=1)),
            "daily_sessions": len(daily),
            "direction": direction,
            "direction_auc": _auc(target[active] > 0, prediction[active]),
            **_cross_sectional(indices, target, prediction),
        }
        item["calibration_intercept_target_on_prediction"] = float(
            target.mean()
            - item["calibration_slope_target_on_prediction"] * prediction.mean()
        )
        by_horizon[str(horizon)] = item
    items = [by_horizon[str(horizon)] for horizon in HORIZONS_V1]
    summary = {
        "best_step": int(best["global_step"]),
        "selection_loss_function": loss_function,
        "equal_head_standardized_selection_loss": float(best[selection_key]),
        f"equal_head_standardized_{loss_function.lower()}": float(
            best[selection_key]
        ),
        "equal_head_raw_rmse": float(np.mean([item["rmse"] for item in items])),
        "equal_head_zero_rmse_skill": float(
            np.mean([item["zero_rmse_skill"] for item in items])
        ),
        "equal_head_train_median_rmse_skill": float(
            np.mean([item["train_median_rmse_skill"] for item in items])
        ),
        "equal_head_direction_accuracy": float(
            np.mean([item["direction"]["accuracy"] for item in items])
        ),
        "equal_head_majority_accuracy": float(
            np.mean([item["direction"]["majority_accuracy"] for item in items])
        ),
        "equal_head_accuracy_excess_vs_majority": float(
            np.mean(
                [item["direction"]["accuracy_excess_vs_majority"] for item in items]
            )
        ),
        "equal_head_balanced_accuracy": float(
            np.mean([item["direction"]["balanced_accuracy"] for item in items])
        ),
        "equal_head_mcc": float(np.mean([item["direction"]["mcc"] for item in items])),
        "equal_head_auc": float(np.mean([item["direction_auc"] for item in items])),
        "equal_head_pearson": float(np.mean([item["pearson"] for item in items])),
        "equal_head_rank_ic": float(np.mean([item["rank_ic_mean"] for item in items])),
        "equal_head_top_bottom_spread": float(
            np.mean([item["top_bottom_spread_mean"] for item in items])
        ),
    }
    return {
        "schema_version": "courage_strict_continuous_origin_valid_metrics_v1",
        "role": "rolling_valid_checkpoint_selection_population",
        "is_unseen_test": False,
        "checkpoint_selection_loss_function": loss_function,
        "checkpoint_selection_uses_only_declared_standardized_loss": True,
        "summary": summary,
        "by_horizon": by_horizon,
    }


def _render(origin: str, metrics: dict[str, Any]) -> str:
    summary = metrics["summary"]
    loss_name = summary["selection_loss_function"]
    lines = [
        f"# Courage Continuous {origin} 完整 Valid 评测",
        "",
        "> 这是滚动 Valid/checkpoint 选择人口，不是未见 Test；所有 horizon 均使用各自的成熟 Label mask。",
        "",
        "## 汇总",
        "",
        f"- 最佳 checkpoint：step `{summary['best_step']}`；标准化 {loss_name}：`{summary['equal_head_standardized_selection_loss']:.9f}`。",
        f"- 等权 raw RMSE：`{summary['equal_head_raw_rmse']:.8f}`；相对零预测 skill：`{summary['equal_head_zero_rmse_skill']:+.3%}`；相对 Train 中位数常数 skill：`{summary['equal_head_train_median_rmse_skill']:+.3%}`。",
        f"- 等权 ACC：`{summary['equal_head_direction_accuracy']:.3%}`；多数类：`{summary['equal_head_majority_accuracy']:.3%}`；超额：`{summary['equal_head_accuracy_excess_vs_majority']:+.3%}`。",
        f"- 等权 BAcc：`{summary['equal_head_balanced_accuracy']:.3%}`；MCC：`{summary['equal_head_mcc']:+.5f}`；AUC：`{summary['equal_head_auc']:.3%}`。",
        f"- 等权 Pearson：`{summary['equal_head_pearson']:+.5f}`；Rank IC：`{summary['equal_head_rank_ic']:+.5f}`；Top−Bottom：`{summary['equal_head_top_bottom_spread']:+.6f}`。",
        "",
        "## 逐 horizon",
        "",
        "| H | Rows | RMSE | Zero skill | Median skill | ACC | Majority | BAcc | MCC | AUC | Pearson | Rank IC | Spread |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS_V1:
        item = metrics["by_horizon"][str(horizon)]
        direction = item["direction"]
        lines.append(
            f"| {horizon} | {item['rows']:,} | {item['rmse']:.8f} | {item['zero_rmse_skill']:+.3%} | "
            f"{item['train_median_rmse_skill']:+.3%} | {direction['accuracy']:.3%} | "
            f"{direction['majority_accuracy']:.3%} | {direction['balanced_accuracy']:.3%} | "
            f"{direction['mcc']:+.5f} | {item['direction_auc']:.3%} | {item['pearson']:+.5f} | "
            f"{item['rank_ic_mean']:+.5f} | {item['top_bottom_spread_mean']:+.6f} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "- 收益回归值直接作为方向分数计算 AUC，并非上涨概率。",
        "- ACC 排除真实收益恰好为 0 的样本，且必须与同人口多数类、BAcc 和 MCC 一起解释。",
        "- Rank IC/Spread 仅为诊断，不参与 checkpoint 选择。",
        "- 未读取 April、May、June 或更晚数据；未执行 refit、策略、回测、交易或远端推送。",
        "",
    ]
    return "\n".join(lines)


def evaluate(origin: str, *, config_path: Path = CONFIG) -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world != 8 or torch.cuda.device_count() != 8:
        raise ContinuousOriginEvaluationError(
            "evaluation requires exact 8-GPU torchrun"
        )
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    config = load_config(config_path)
    identity = validate_identity(config)
    if origin not in config.origins:
        raise ContinuousOriginEvaluationError(f"unknown origin: {origin}")
    origin_root = config.output_root / origin
    checkpoint_path = origin_root / "best.pt"
    manifest_path = origin_root / "_training_manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise ContinuousOriginEvaluationError("origin training is not complete")
    training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        training_manifest.get("decision") != "PASS_CONTINUOUS_ORIGIN_TRAINING_COMPLETE"
        or checkpoint.get("schema_version") != "courage_strict_continuous_checkpoint_v1"
        or checkpoint.get("experiment_id") != config.source["experiment_id"]
        or checkpoint.get("origin") != origin
        or checkpoint.get("input_identity") != identity
        or sha256_file(checkpoint_path)
        != training_manifest.get("best_checkpoint_sha256")
    ):
        raise ContinuousOriginEvaluationError("training/checkpoint identity drift")
    segment = tuple(config.origins[origin]["valid"])
    dataset = CourageStrictV1Dataset(
        provider_root=config.provider_root,
        segments={"valid": segment},
        turnover_band=(
            tuple(config.source["membership_candidate"]["turnover_band"])
            if "membership_candidate" in config.source
            else None
        ),
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
            target = batch["raw_targets"].numpy().astype(np.float32, copy=False)
            mask = batch["target_mask"].numpy().astype(bool, copy=False)
            calendar_indices.append(batch["calendar_index"].numpy())
            symbol_positions.append(batch["symbol_position"].numpy())
            moved = _move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = _forward(model, moved).float().cpu().numpy()
            for head, horizon in enumerate(HORIZONS_V1):
                stats = checkpoint["scalers"]["targets"][str(horizon)]
                prediction[:, head] = (
                    prediction[:, head] * stats["iqr"] + stats["median"]
                )
            predictions.append(prediction)
            targets.append(target)
            masks.append(mask)
    predicted = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
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
        raise ContinuousOriginEvaluationError("rank population drift")
    output = origin_root / "valid_evaluation"
    if rank == 0:
        if output.exists():
            raise ContinuousOriginEvaluationError(f"output already exists: {output}")
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
            raise ContinuousOriginEvaluationError("merged row order drift")
        best = json.loads((origin_root / "best.json").read_text(encoding="utf-8"))
        raw = scaled.raw
        metrics = _metrics(
            merged,
            scalers=checkpoint["scalers"],
            best=best,
            calendar=raw.provider.calendar,
        )
        metrics.update(
            {
                "experiment_id": config.source["experiment_id"],
                "origin": origin,
                "valid_segment": list(segment),
                "population_rows": len(merged),
                "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "read_before_exclusive": segment[1],
            }
        )
        merged.to_parquet(
            output / "valid_predictions_and_targets.parquet",
            index=False,
            compression="zstd",
        )
        atomic_json(output / "valid_metrics.json", metrics)
        report = _render(origin, metrics)
        (output / "VALID_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
        docs_root = ROOT / "docs" / config.source["experiment_id"]
        docs_root.mkdir(parents=True, exist_ok=True)
        (docs_root / f"{origin.upper()}_VALID_EVALUATION.md").write_text(
            report, encoding="utf-8"
        )
        manifest = {
            "schema_version": "courage_strict_continuous_origin_evaluation_manifest_v1",
            "decision": "PASS_COMPLETE_ROLLING_VALID_EVALUATION",
            "origin": origin,
            "population_rows": len(merged),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics_sha256": sha256_file(output / "valid_metrics.json"),
            "predictions_sha256": sha256_file(
                output / "valid_predictions_and_targets.parquet"
            ),
            "report_sha256": sha256_file(output / "VALID_EVALUATION_REPORT.md"),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "is_unseen_test": False,
            "april_read_executed": False,
            "may_read_executed": False,
            "june_or_later_read_executed": False,
            "refit_executed": False,
            "strategy_or_backtest_executed": False,
            "trading_executed": False,
            "remote_push_executed": False,
        }
        atomic_json(output / "_evaluation_manifest.json", manifest)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin",
        choices=("origin_2026_01", "origin_2026_02", "origin_2026_03"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    evaluate(args.origin, config_path=args.config.resolve())


if __name__ == "__main__":
    main()
