"""Evaluation diagnostics for Courage Strict V1 predictions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from qlib.contrib.data.courage_strict_v1.dataset import CourageStrictV1TorchDataset
from qlib.contrib.model.patchtst_courage_strict_v1 import HORIZONS_V1


class CourageStrictV1EvaluationError(RuntimeError):
    """Raised when predictions and the frozen evaluation population diverge."""


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
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "active_rows": count,
        "flat_target_rows": int((target == 0).sum()),
        "accuracy": (tp + tn) / count if count else math.nan,
        "majority_accuracy": majority,
        "balanced_accuracy": (recall_up + recall_down) / 2,
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0,
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
    timestamp: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {"timestamp": timestamp, "target": target, "prediction": prediction}
    )
    grouped = frame.groupby("timestamp", sort=False)
    values = grouped.apply(
        lambda group: group["prediction"].corr(group["target"], method="spearman"),
        include_groups=False,
    )
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    frame["prediction_rank"] = grouped["prediction"].rank(method="average", pct=True)
    top = frame.loc[frame.prediction_rank >= 0.9].groupby("timestamp").target.mean()
    bottom = frame.loc[frame.prediction_rank <= 0.1].groupby("timestamp").target.mean()
    spread = top.subtract(bottom, fill_value=np.nan).dropna()
    return {
        "rank_ic_mean": float(values.mean()) if len(values) else math.nan,
        "rank_ic_std_ddof1": float(values.std(ddof=1)),
        "rank_ic_timestamps": int(len(values)),
        "top_bottom_spread_mean": float(spread.mean()) if len(spread) else math.nan,
        "top_bottom_timestamps": int(len(spread)),
    }


def load_targets(
    raw: CourageStrictV1TorchDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.empty((len(raw), len(HORIZONS_V1)), dtype=np.float32)
    masks = np.empty_like(targets, dtype=bool)
    cursor = 0
    for position in np.unique(raw.symbol_positions):
        selected = raw.symbol_positions == position
        ends = raw.ends[selected]
        arrays = raw._arrays(int(position))
        count = len(ends)
        for head, horizon in enumerate(HORIZONS_V1):
            values = np.asarray(arrays[f"label_return_{horizon}"][ends])
            valid = np.asarray(arrays[f"label_valid_{horizon}"][ends]) > 0.5
            targets[cursor : cursor + count, head] = values
            masks[cursor : cursor + count, head] = valid
        cursor += count
    if cursor != len(raw):
        raise CourageStrictV1EvaluationError("target population order drift")
    return targets, masks, raw.ends.copy()


def evaluate_predictions(
    *,
    raw: CourageStrictV1TorchDataset,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    if len(predictions) != len(raw):
        raise CourageStrictV1EvaluationError("prediction population size drift")
    targets, masks, calendar_indices = load_targets(raw)
    expected_columns = [f"prediction_{horizon}" for horizon in HORIZONS_V1]
    if list(predictions.columns) != expected_columns:
        raise CourageStrictV1EvaluationError("prediction head order drift")
    predicted = predictions.to_numpy(dtype=np.float64)
    timestamp = raw.provider.calendar[calendar_indices].to_numpy()
    by_horizon: dict[str, Any] = {}
    for head, horizon in enumerate(HORIZONS_V1):
        valid = masks[:, head]
        target = targets[valid, head].astype(np.float64)
        prediction = predicted[valid, head]
        if (
            not len(target)
            or not np.isfinite(target).all()
            or not np.isfinite(prediction).all()
        ):
            raise CourageStrictV1EvaluationError(
                f"invalid evaluation values: {horizon}"
            )
        error = prediction - target
        centered_target = target - target.mean()
        centered_prediction = prediction - prediction.mean()
        pearson_den = math.sqrt(
            float(
                np.square(centered_target).sum() * np.square(centered_prediction).sum()
            )
        )
        cross_sectional = _cross_sectional(timestamp[valid], target, prediction)
        daily = (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(timestamp[valid]).date,
                    "squared_error": np.square(error),
                }
            )
            .groupby("date")["squared_error"]
            .mean()
            .pow(0.5)
        )
        by_horizon[str(horizon)] = {
            "rows": int(len(target)),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()),
            "pearson": (
                float((centered_target * centered_prediction).sum() / pearson_den)
                if pearson_den
                else 0.0
            ),
            **cross_sectional,
            "daily_rmse_mean": float(daily.mean()),
            "daily_rmse_std_ddof1": float(daily.std(ddof=1)),
            "daily_sessions": int(len(daily)),
            "direction": _direction(target, prediction),
        }
    return {
        "schema_version": "courage_strict_v1_evaluation_metrics_v1",
        "role": "valid",
        "primary_metrics": ["rmse", "mae", "bias", "pearson"],
        "diagnostic_metrics": [
            "rank_ic_mean",
            "direction.accuracy",
            "direction.balanced_accuracy",
            "direction.mcc",
        ],
        "ordinary_ACC_is_model_gate": False,
        "by_horizon": by_horizon,
    }


def render_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Courage Strict V1 Valid 评测报告",
        "",
        "本报告由新 Qlib V1 的冻结 Valid 人口生成。RMSE/MAE/bias/Pearson 为主要诊断；ACC、"
        "balanced ACC、MCC、Rank IC 和 Top-Bottom spread 不参与 checkpoint 选择。",
        "",
        "| Horizon | Rows | RMSE | MAE | Bias | Pearson | Rank IC | Top-Bottom | ACC | Majority | BAcc | MCC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS_V1:
        item = metrics["by_horizon"][str(horizon)]
        direction = item["direction"]
        lines.append(
            "| {h} | {rows:,} | {rmse:.8f} | {mae:.8f} | {bias:.8f} | "
            "{pearson:.6f} | {rank:.6f} | {spread:.8f} | {acc:.4%} | "
            "{majority:.4%} | {bacc:.4%} | {mcc:.6f} |".format(
                h=horizon,
                rows=item["rows"],
                rmse=item["rmse"],
                mae=item["mae"],
                bias=item["bias"],
                pearson=item["pearson"],
                rank=item["rank_ic_mean"],
                spread=item["top_bottom_spread_mean"],
                acc=direction["accuracy"],
                majority=direction["majority_accuracy"],
                bacc=direction["balanced_accuracy"],
                mcc=direction["mcc"],
            )
        )
    lines += [
        "",
        "方向 ACC 排除真实收益恰好为 0 的样本；预测为 0 对非零真实收益计错。Majority 是同一"
        "评测人口的多数类基线，因此不能只看裸 ACC。",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "CourageStrictV1EvaluationError",
    "evaluate_predictions",
    "load_targets",
    "render_markdown",
]
