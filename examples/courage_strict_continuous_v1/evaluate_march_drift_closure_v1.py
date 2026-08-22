"""Close March drift diagnostics from the frozen Continuous V1 predictions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.courage_strict_continuous_v1.evaluate_baseline_closure_v1 import (
    FEATURES,
    _quantiles,
    _timestamp_rank_ic,
    metric_row,
)
from qlib.contrib.data.courage_strict_v1.dataset import (
    CourageStrictV1Dataset,
    label_maturity_mask_v1,
)
from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c4_features_v1 import (
    DYNAMIC_FEATURES,
    SLOW_FEATURES,
)
from qlib.contrib.model.patchtst_courage_strict_v1 import HORIZONS_V1

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "examples/courage_strict_continuous_v1/march_drift_diagnostic_contract_v1.json"
)


class MarchDriftError(RuntimeError):
    """Raised when a frozen input, population, or boundary drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(contract: dict[str, Any], key: str) -> Path:
    return ROOT / contract[key]


def validate_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version")
        != "courage_strict_continuous_march_drift_contract_v1"
        or contract.get("source_experiment_id") != "courage_strict_continuous_v1"
        or contract["scope"].get("read_before_exclusive") != "2026-04-01"
        or contract["scope"].get("april_or_later_read") is not False
    ):
        raise MarchDriftError("March drift contract identity/boundary drift")
    for key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("provider_catalog", "provider_catalog_sha256"),
        ("train_scalers", "train_scalers_sha256"),
        ("january_predictions", "january_predictions_sha256"),
        ("february_predictions", "february_predictions_sha256"),
        ("march_predictions", "march_predictions_sha256"),
        ("evaluator", "evaluator_sha256"),
    ):
        path = _resolve(contract, key)
        if not path.is_file() or sha256_file(path) != contract[hash_key]:
            raise MarchDriftError(f"frozen input drift: {key}")
    forbidden = set(contract["authority"]) - {"read_existing_pre_april_data"}
    if contract["authority"].get("read_existing_pre_april_data") is not True or any(
        contract["authority"][key] is not False for key in forbidden
    ):
        raise MarchDriftError("forbidden authority drift")
    return contract


def _extract_population(raw: Any) -> dict[str, np.ndarray]:
    rows = len(raw)
    terminal = np.empty((rows, len(FEATURES)), dtype=np.float32)
    usable = np.empty((rows, len(FEATURES)), dtype=bool)
    targets = np.zeros((rows, len(HORIZONS_V1)), dtype=np.float32)
    masks = np.zeros((rows, len(HORIZONS_V1)), dtype=bool)
    minute_slot = np.empty(rows, dtype=np.int16)
    industry_id = np.empty(rows, dtype=np.int16)
    turnover = np.full(rows, np.nan, dtype=np.float32)
    positions = raw.symbol_positions
    boundaries = np.r_[0, np.flatnonzero(np.diff(positions)) + 1, rows]
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        position = int(positions[left])
        ends = raw.ends[left:right].astype(np.int64, copy=False)
        arrays = raw._arrays(position)
        for feature_index, feature in enumerate(FEATURES):
            values = np.asarray(arrays[feature][ends], dtype=np.float32)
            feature_usable = (
                (arrays[f"{feature}__available"][ends] > 0.5)
                & (arrays[f"{feature}__data_missing"][ends] <= 0.5)
                & np.isfinite(values)
            )
            terminal[left:right, feature_index] = values
            usable[left:right, feature_index] = feature_usable
        minute_slot[left:right] = np.asarray(
            arrays["minute_slot"][ends], dtype=np.int16
        )
        industry_id[left:right] = np.asarray(
            arrays["industry_id"][ends], dtype=np.int16
        )
        turnover_index = len(DYNAMIC_FEATURES) + SLOW_FEATURES.index("turnover_mean_60")
        turnover[left:right] = np.where(
            usable[left:right, turnover_index],
            terminal[left:right, turnover_index],
            np.nan,
        )
        for head, horizon in enumerate(HORIZONS_V1):
            values = np.asarray(
                arrays[f"label_return_{horizon}"][ends], dtype=np.float32
            )
            active = label_maturity_mask_v1(
                (arrays[f"label_valid_{horizon}"][ends] > 0.5) & np.isfinite(values),
                arrays[f"label_target_end_index_{horizon}"][ends],
                arrays[f"label_available_index_{horizon}"][ends],
                cutoff_index=raw.label_cutoff_index,
            )
            targets[left:right, head] = np.where(active, values, 0.0)
            masks[left:right, head] = active
    if not np.isfinite(terminal[usable]).all():
        raise MarchDriftError("usable terminal feature is non-finite")
    return {
        "terminal": terminal,
        "usable": usable,
        "targets": targets,
        "masks": masks,
        "minute_slot": minute_slot,
        "industry_id": industry_id,
        "turnover": turnover,
        "symbol_position": raw.symbol_positions.copy(),
        "calendar_index": raw.ends.copy(),
    }


def _load_predictions(path: Path, population: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_values("row_index", kind="stable")
    rows = len(population["calendar_index"])
    if (
        len(frame) != rows
        or not np.array_equal(frame.row_index.to_numpy(), np.arange(rows))
        or not np.array_equal(
            frame.calendar_index.to_numpy(dtype=np.int64),
            population["calendar_index"].astype(np.int64),
        )
        or not np.array_equal(
            frame.symbol_position.to_numpy(dtype=np.int32),
            population["symbol_position"].astype(np.int32),
        )
    ):
        raise MarchDriftError("prediction population key drift")
    for head, horizon in enumerate(HORIZONS_V1):
        observed_mask = frame[f"valid_{horizon}"].to_numpy(dtype=bool)
        observed_target = frame[f"target_{horizon}"].to_numpy(dtype=np.float32)
        active = population["masks"][:, head]
        if not np.array_equal(observed_mask, active) or not np.array_equal(
            observed_target[active], population["targets"][active, head]
        ):
            raise MarchDriftError(f"prediction target/mask drift: {horizon}")
    return frame


def _direction_majority(target: np.ndarray) -> float:
    active = target != 0
    truth = target[active] > 0
    return max(float(truth.mean()), float((~truth).mean()))


def _period_metrics(
    split: str,
    population: dict[str, np.ndarray],
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    records = []
    for head, horizon in enumerate(HORIZONS_V1):
        active = population["masks"][:, head]
        target = population["targets"][active, head].astype(np.float64)
        prediction = predictions.loc[active, f"prediction_{horizon}"].to_numpy(
            dtype=np.float64
        )
        item = metric_row(target, prediction)
        timestamp_ic = _timestamp_rank_ic(
            pd.DataFrame(
                {
                    "calendar_index": population["calendar_index"][active],
                    "target": target,
                    "prediction": prediction,
                }
            )
        )
        zero_rmse = float(np.sqrt(np.square(target).mean()))
        majority = _direction_majority(target)
        records.append(
            {
                "split": split,
                "horizon": int(horizon),
                **item,
                "timestamp_rank_ic_mean": float(timestamp_ic.mean()),
                "zero_rmse_skill": 1.0 - item["rmse"] / zero_rmse,
                "majority_accuracy": majority,
                "accuracy_excess_vs_majority": item["accuracy"] - majority,
            }
        )
    return records


def _march_daily_metrics(
    population: dict[str, np.ndarray],
    predictions: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    records = []
    for head, horizon in enumerate(HORIZONS_V1):
        active = population["masks"][:, head]
        indices = population["calendar_index"][active].astype(np.int64)
        dates = pd.to_datetime(calendar[indices]).normalize()
        frame = pd.DataFrame(
            {
                "date": dates,
                "calendar_index": indices,
                "target": population["targets"][active, head].astype(np.float64),
                "prediction": predictions.loc[active, f"prediction_{horizon}"].to_numpy(
                    dtype=np.float64
                ),
            }
        )
        timestamp_ic = _timestamp_rank_ic(frame)
        ic_dates = pd.to_datetime(
            calendar[timestamp_ic.index.to_numpy(dtype=np.int64)]
        ).normalize()
        daily_ic = (
            pd.Series(timestamp_ic.to_numpy(), index=ic_dates).groupby(level=0).mean()
        )
        for date, group in frame.groupby("date", sort=True):
            item = metric_row(group.target.to_numpy(), group.prediction.to_numpy())
            majority = _direction_majority(group.target.to_numpy())
            records.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "horizon": int(horizon),
                    **item,
                    "timestamp_rank_ic_mean": float(daily_ic.get(date, math.nan)),
                    "majority_accuracy": majority,
                    "accuracy_excess_vs_majority": item["accuracy"] - majority,
                }
            )
    return pd.DataFrame(records)


def _march_grouped_metrics(
    population: dict[str, np.ndarray],
    predictions: pd.DataFrame,
    *,
    minimum_rows: int,
) -> pd.DataFrame:
    slot = population["minute_slot"].astype(np.int64)
    minute_bucket = np.array(
        [f"{value // 30 * 30:03d}-{value // 30 * 30 + 29:03d}" for value in slot]
    )
    turnover_band = pd.cut(
        population["turnover"],
        bins=[5.0, 7.5, 10.0, 12.5, 15.000001],
        labels=["5-7.5", "7.5-10", "10-12.5", "12.5-15"],
        include_lowest=True,
    ).astype(str)
    dimensions = {
        "minute_30slot_bucket": minute_bucket,
        "industry_id": population["industry_id"].astype(str),
        "turnover_band": np.asarray(turnover_band),
    }
    records = []
    for dimension, values in dimensions.items():
        groups = pd.Series(np.arange(len(values))).groupby(values, sort=True).indices
        for group, raw_indices in groups.items():
            selected = np.asarray(raw_indices, dtype=np.int64)
            for head, horizon in enumerate(HORIZONS_V1):
                active = population["masks"][selected, head]
                if int(active.sum()) < minimum_rows:
                    continue
                indices = selected[active]
                target = population["targets"][indices, head].astype(np.float64)
                prediction = predictions.iloc[indices][
                    f"prediction_{horizon}"
                ].to_numpy(dtype=np.float64)
                item = metric_row(target, prediction)
                majority = _direction_majority(target)
                records.append(
                    {
                        "dimension": dimension,
                        "group": str(group),
                        "horizon": int(horizon),
                        **item,
                        "majority_accuracy": majority,
                        "accuracy_excess_vs_majority": item["accuracy"] - majority,
                    }
                )
    return pd.DataFrame(records)


def _feature_reference(
    population: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    references = {}
    records = []
    for index, feature in enumerate(FEATURES):
        active = population["usable"][:, index]
        values = population["terminal"][active, index].astype(np.float64)
        quantile_grid = np.quantile(values, np.linspace(0.01, 0.99, 99))
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 11)))
        if len(edges) < 3:
            center = float(np.median(values))
            edges = np.array([center - 1e-12, center, center + 1e-12])
        edges[0] = -np.inf
        edges[-1] = np.inf
        histogram = np.histogram(values, bins=edges)[0].astype(np.float64)
        histogram /= histogram.sum()
        stats = _quantiles(values)
        records.append(
            {
                "split": "train",
                "feature": feature,
                "feature_group": "dynamic" if index < len(DYNAMIC_FEATURES) else "slow",
                "population_rows": len(active),
                "missing_rate": float((~active).mean()),
                "missing_rate_delta_vs_train": 0.0,
                "median_shift_train_iqr": 0.0,
                "iqr_ratio_vs_train": 1.0,
                "psi_vs_train": 0.0,
                "quantile_w1_vs_train": 0.0,
                **stats,
            }
        )
        references[feature] = {
            "quantiles": quantile_grid,
            "edges": edges,
            "histogram": histogram,
            "median": np.array([stats["median"]]),
            "iqr": np.array([stats["iqr"]]),
            "missing_rate": np.array([float((~active).mean())]),
        }
    return references, records


def _feature_drift(
    split: str,
    population: dict[str, np.ndarray],
    references: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    records = []
    epsilon = 1e-8
    for index, feature in enumerate(FEATURES):
        active = population["usable"][:, index]
        values = population["terminal"][active, index].astype(np.float64)
        stats = _quantiles(values)
        reference = references[feature]
        histogram = np.histogram(values, bins=reference["edges"])[0].astype(np.float64)
        histogram /= histogram.sum()
        left = np.clip(reference["histogram"], epsilon, None)
        right = np.clip(histogram, epsilon, None)
        psi = float(np.sum((right - left) * np.log(right / left)))
        quantiles = np.quantile(values, np.linspace(0.01, 0.99, 99))
        train_iqr = float(reference["iqr"][0])
        records.append(
            {
                "split": split,
                "feature": feature,
                "feature_group": "dynamic" if index < len(DYNAMIC_FEATURES) else "slow",
                "population_rows": len(active),
                "missing_rate": float((~active).mean()),
                "missing_rate_delta_vs_train": float(
                    (~active).mean() - reference["missing_rate"][0]
                ),
                "median_shift_train_iqr": float(
                    (stats["median"] - reference["median"][0]) / train_iqr
                ),
                "iqr_ratio_vs_train": float(stats["iqr"] / train_iqr),
                "psi_vs_train": psi,
                "quantile_w1_vs_train": float(
                    np.mean(np.abs(quantiles - reference["quantiles"]))
                ),
                **stats,
            }
        )
    return records


def _distribution_records(
    split: str,
    population: dict[str, np.ndarray],
    predictions: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    records = []
    for head, horizon in enumerate(HORIZONS_V1):
        active = population["masks"][:, head]
        records.append(
            {
                "split": split,
                "kind": "target",
                "horizon": int(horizon),
                **_quantiles(population["targets"][active, head]),
            }
        )
        if predictions is not None:
            records.append(
                {
                    "split": split,
                    "kind": "prediction",
                    "horizon": int(horizon),
                    **_quantiles(
                        predictions.loc[active, f"prediction_{horizon}"].to_numpy()
                    ),
                }
            )
    return records


def _systematic_indices(rows: int, maximum: int) -> np.ndarray:
    selected = min(rows, maximum)
    indices = np.floor((np.arange(selected) + 0.5) * rows / selected).astype(np.int64)
    if len(np.unique(indices)) != selected or indices[0] < 0 or indices[-1] >= rows:
        raise MarchDriftError("lookback systematic sample drift")
    return indices


def _lookback_profiles(
    raw_by_split: dict[str, Any],
    *,
    maximum_samples: int,
    age_bins: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_by_split: dict[str, dict[int, np.ndarray]] = {}
    selected_rows = {}
    for split, raw in raw_by_split.items():
        selected = _systematic_indices(len(raw), maximum_samples)
        selected_rows[split] = len(selected)
        positions = raw.symbol_positions[selected]
        ends = raw.ends[selected]
        grouped = {}
        for position in np.unique(positions):
            grouped[int(position)] = ends[positions == position].astype(np.int64)
        selected_by_split[split] = grouped
    profile = defaultdict(
        lambda: {"sequences": 0, "slots": 0, "usable_slots": 0, "fully_missing": 0}
    )
    trailing: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    positions = sorted(
        set().union(*(set(grouped) for grouped in selected_by_split.values()))
    )
    provider = next(iter(raw_by_split.values()))
    for position in positions:
        arrays = provider._arrays(position)
        axis = np.arange(len(arrays[DYNAMIC_FEATURES[0]]), dtype=np.int64)
        for feature in DYNAMIC_FEATURES:
            values = np.asarray(arrays[feature])
            usable = (
                (arrays[f"{feature}__available"] > 0.5)
                & (arrays[f"{feature}__data_missing"] <= 0.5)
                & np.isfinite(values)
            )
            prefix = np.r_[0, np.cumsum(usable, dtype=np.int64)]
            last_usable = np.maximum.accumulate(np.where(usable, axis, -1))
            for split, grouped in selected_by_split.items():
                if position not in grouped:
                    continue
                ends = grouped[position]
                if int(ends.min()) < max(high for _, high in age_bins):
                    raise MarchDriftError(
                        "lookback sample lacks full 1200-minute history"
                    )
                trailing[(split, feature)].append(
                    np.minimum(ends - last_usable[ends], 1200).astype(np.int16)
                )
                for low, high in age_bins:
                    left = ends - high
                    right = ends - low
                    counts = prefix[right + 1] - prefix[left]
                    key = (split, feature, low, high)
                    profile[key]["sequences"] += len(ends)
                    profile[key]["slots"] += len(ends) * (high - low + 1)
                    profile[key]["usable_slots"] += int(counts.sum())
                    profile[key]["fully_missing"] += int((counts == 0).sum())
    profile_rows = []
    for (split, feature, low, high), values in sorted(profile.items()):
        profile_rows.append(
            {
                "split": split,
                "feature": feature,
                "age_low_minutes": low,
                "age_high_minutes": high,
                "sampled_sequences": values["sequences"],
                "usable_fraction": values["usable_slots"] / values["slots"],
                "fully_missing_sequence_rate": values["fully_missing"]
                / values["sequences"],
            }
        )
    trailing_rows = []
    for (split, feature), chunks in sorted(trailing.items()):
        values = np.concatenate(chunks).astype(np.int64)
        trailing_rows.append(
            {
                "split": split,
                "feature": feature,
                "sampled_sequences": len(values),
                "mean_trailing_missing_minutes": float(values.mean()),
                "p95_trailing_missing_minutes": float(np.quantile(values, 0.95)),
                "p99_trailing_missing_minutes": float(np.quantile(values, 0.99)),
                "max_trailing_missing_minutes": int(values.max()),
            }
        )
    if any(
        row["sampled_sequences"] != selected_rows[row["split"]] for row in trailing_rows
    ):
        raise MarchDriftError("lookback sample coverage drift")
    return pd.DataFrame(profile_rows), pd.DataFrame(trailing_rows)


def _render_figures(
    output: Path,
    distributions: pd.DataFrame,
    feature_drift: pd.DataFrame,
    lookback: pd.DataFrame,
) -> None:
    target = distributions.loc[distributions.kind.eq("target")]
    figure, axis = plt.subplots(figsize=(11, 6))
    for split in ("train", "january", "february", "march"):
        selected = target.loc[target.split.eq(split)].sort_values("horizon")
        axis.plot(selected.horizon, selected.up_rate, marker="o", label=split)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Horizon (minutes)")
    axis.set_ylabel("Target up rate")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "TARGET_UP_RATE_DRIFT.png", dpi=180)
    plt.close(figure)

    matrix = (
        feature_drift.loc[feature_drift.split.ne("train")]
        .pivot(index="feature", columns="split", values="median_shift_train_iqr")
        .reindex(columns=["january", "february", "march"])
    )
    figure, axis = plt.subplots(figsize=(8, 9))
    image = axis.imshow(
        matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1
    )
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("Feature median shift / Train IQR")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "FEATURE_MEDIAN_DRIFT.png", dpi=180)
    plt.close(figure)

    summary = (
        lookback.groupby(["split", "age_low_minutes", "age_high_minutes"], sort=True)
        .usable_fraction.mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    for split in ("train", "january", "february", "march"):
        selected = summary.loc[summary.split.eq(split)]
        labels = [
            f"{low}-{high}"
            for low, high in zip(
                selected.age_low_minutes, selected.age_high_minutes, strict=True
            )
        ]
        axis.plot(labels, selected.usable_fraction, marker="o", label=split)
    axis.set_xlabel("Lookback age bin (minutes)")
    axis.set_ylabel("Mean dynamic-feature usable fraction")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "LOOKBACK_USABLE_FRACTION.png", dpi=180)
    plt.close(figure)


def _render_report(
    period: pd.DataFrame,
    daily: pd.DataFrame,
    grouped: pd.DataFrame,
    features: pd.DataFrame,
    lookback: pd.DataFrame,
    trailing: pd.DataFrame,
) -> str:
    march = period.loc[period.split.eq("march")].sort_values("horizon")
    top_feature = (
        features.loc[features.split.eq("march")]
        .sort_values("psi_vs_train", ascending=False)
        .head(5)
    )
    lookback_summary = lookback.groupby("split").usable_fraction.mean().to_dict()
    trailing_summary = (
        trailing.groupby("split").p95_trailing_missing_minutes.mean().to_dict()
    )
    march_daily = daily.groupby("horizon").agg(
        days=("date", "nunique"),
        target_up_rate=("target_up_rate", "mean"),
        prediction_up_rate=("prediction_up_rate", "mean"),
        bacc=("balanced_accuracy", "mean"),
        mcc=("mcc", "mean"),
        rank_ic=("timestamp_rank_ic_mean", "mean"),
    )
    lines = [
        "# Courage Strict Continuous V1 March漂移诊断闭环",
        "",
        "## 结论",
        "",
        "- 本报告使用原V1冻结step2250及既有January—March预测，不重新训练、不重新选checkpoint。",
        f"- March各horizon真实上涨率范围为`{march.target_up_rate.min():.1%}`—`{march.target_up_rate.max():.1%}`，多数方向基线因此升至约`{march.majority_accuracy.min():.1%}`—`{march.majority_accuracy.max():.1%}`。",
        f"- March模型prediction up rate范围为`{march.prediction_up_rate.min():.1%}`—`{march.prediction_up_rate.max():.1%}`；绝对方向与当月Label基准明显错位。",
        "- 240/480m仍保留正Rank IC，但这与绝对收益均值和方向失配同时存在，支持“相对排序尚存、绝对条件均值漂移”的判断。",
        f"- 动态lookback平均可用率Train/January/February/March分别为`{lookback_summary['train']:.1%}`/`{lookback_summary['january']:.1%}`/`{lookback_summary['february']:.1%}`/`{lookback_summary['march']:.1%}`；名义1200分钟窗口并未因大面积缺失而整体失效。",
        "- 当前模型仍保持`FAIL_BASELINE_GATE`；本诊断不授权April或新训练。",
        "",
        "## March逐horizon汇总",
        "",
        "| H | Target up | Pred up | ACC | Majority | ACC excess | BAcc | MCC | AUC | Rank IC | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in march.itertuples(index=False):
        lines.append(
            f"| {int(row.horizon)} | {row.target_up_rate:.1%} | {row.prediction_up_rate:.1%} | "
            f"{row.accuracy:.1%} | {row.majority_accuracy:.1%} | {row.accuracy_excess_vs_majority:+.1%} | "
            f"{row.balanced_accuracy:.1%} | {row.mcc:+.4f} | {row.auc:.1%} | "
            f"{row.timestamp_rank_ic_mean:+.4f} | {row.bias:+.3%} |"
        )
    lines += [
        "",
        "## March逐日摘要",
        "",
        "| H | Days | Mean target up | Mean pred up | Mean BAcc | Mean MCC | Mean daily Rank IC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, row in march_daily.iterrows():
        lines.append(
            f"| {int(horizon)} | {int(row.days)} | {row.target_up_rate:.1%} | "
            f"{row.prediction_up_rate:.1%} | {row.bacc:.1%} | {row.mcc:+.4f} | {row.rank_ic:+.4f} |"
        )
    lines += [
        "",
        "## 特征漂移最大的5项",
        "",
        "| Feature | Missing Δ | Median shift/Train IQR | IQR ratio | PSI | Quantile-W1 |",
        "|:--|---:|---:|---:|---:|---:|",
    ]
    for row in top_feature.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.missing_rate_delta_vs_train:+.2%} | "
            f"{row.median_shift_train_iqr:+.3f} | {row.iqr_ratio_vs_train:.3f} | "
            f"{row.psi_vs_train:.3f} | {row.quantile_w1_vs_train:.6g} |"
        )
    lines += [
        "",
        "![Target up-rate drift](TARGET_UP_RATE_DRIFT.png)",
        "",
        "![Feature median drift](FEATURE_MEDIAN_DRIFT.png)",
        "",
        "![Lookback usable fraction](LOOKBACK_USABLE_FRACTION.png)",
        "",
        "## 分组与lookback产物",
        "",
        f"- March分组表包含`{len(grouped)}`行，覆盖30分钟位置桶、行业和四个换手率档。",
        f"- Lookback对每个时期固定系统抽样`{int(lookback.sampled_sequences.max()):,}`条序列，并完整扫描每条序列的1200分钟历史。",
        f"- 各时期动态Feature的平均p95末端连续缺失长度为：Train `{trailing_summary['train']:.1f}`、January `{trailing_summary['january']:.1f}`、February `{trailing_summary['february']:.1f}`、March `{trailing_summary['march']:.1f}`分钟。",
        "- `march_grouped_metrics.csv`保留所有满足最小样本数的组，不根据结果筛选。",
        "",
        "## 解释边界",
        "",
        "- PSI和quantile-W1是描述性漂移量，不是因果检验或新模型选择门。",
        "- 行业分组为PIT行业身份；换手率为信号时点可见的T-1慢变量。",
        "- Lookback采用固定系统样本控制高度重叠窗口的重复计数，但每条入样序列覆盖完整1200分钟。",
        "- 未读取2026-04-01及以后数据；未执行训练、refit、策略、回测、交易或远端推送。",
        "",
    ]
    return "\n".join(lines)


def run() -> Path:
    started = time.time()
    contract = validate_contract()
    output = ROOT / contract["outputs"]["artifact_root"]
    doc = ROOT / contract["outputs"]["documentation"]
    if output.exists():
        raise MarchDriftError(f"output already exists: {output}")
    output.mkdir(parents=True)
    provider_root = ROOT / contract["provider_root"]
    segments = {name: tuple(value) for name, value in contract["segments"].items()}
    dataset = CourageStrictV1Dataset(provider_root=provider_root, segments=segments)
    raw_by_split = {
        split: dataset.prepare(split, cache_capacity=8) for split in segments
    }
    age_bins = [tuple(value) for value in contract["lookback"]["age_bins_minutes"]]
    lookback, trailing = _lookback_profiles(
        raw_by_split,
        maximum_samples=int(contract["lookback"]["systematic_samples_per_split"]),
        age_bins=age_bins,
    )
    lookback.to_csv(output / "lookback_missingness.csv", index=False)
    trailing.to_csv(output / "lookback_trailing_missing.csv", index=False)

    prediction_paths = {
        "january": _resolve(contract, "january_predictions"),
        "february": _resolve(contract, "february_predictions"),
        "march": _resolve(contract, "march_predictions"),
    }
    feature_records: list[dict[str, Any]] = []
    distribution_records: list[dict[str, Any]] = []
    period_records: list[dict[str, Any]] = []
    references: dict[str, dict[str, np.ndarray]] | None = None
    march_population = None
    march_predictions = None
    march_calendar = None
    for split in ("train", "january", "february", "march"):
        raw = raw_by_split[split]
        population = _extract_population(raw)
        predictions = (
            _load_predictions(prediction_paths[split], population)
            if split in prediction_paths
            else None
        )
        if split == "train":
            references, initial = _feature_reference(population)
            feature_records.extend(initial)
        else:
            if references is None or predictions is None:
                raise MarchDriftError("Train feature reference missing")
            feature_records.extend(_feature_drift(split, population, references))
            period_records.extend(_period_metrics(split, population, predictions))
        distribution_records.extend(
            _distribution_records(split, population, predictions)
        )
        if split == "march":
            march_population = population
            march_predictions = predictions
            march_calendar = raw.provider.calendar
    if march_population is None or march_predictions is None or march_calendar is None:
        raise MarchDriftError("March diagnostic population absent")
    period = pd.DataFrame(period_records)
    distributions = pd.DataFrame(distribution_records)
    feature_drift = pd.DataFrame(feature_records)
    daily = _march_daily_metrics(march_population, march_predictions, march_calendar)
    grouped = _march_grouped_metrics(
        march_population,
        march_predictions,
        minimum_rows=int(contract["grouping"]["minimum_rows_per_horizon_group"]),
    )
    period.to_csv(output / "period_horizon_metrics.csv", index=False)
    distributions.to_csv(output / "label_prediction_distribution.csv", index=False)
    feature_drift.to_csv(output / "feature_drift.csv", index=False)
    daily.to_csv(output / "march_daily_metrics.csv", index=False)
    grouped.to_csv(output / "march_grouped_metrics.csv", index=False)
    _render_figures(output, distributions, feature_drift, lookback)
    report = _render_report(period, daily, grouped, feature_drift, lookback, trailing)
    (output / "MARCH_DRIFT_DIAGNOSTIC_CLOSURE.md").write_text(report, encoding="utf-8")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(report, encoding="utf-8")
    for name in (
        "TARGET_UP_RATE_DRIFT.png",
        "FEATURE_MEDIAN_DRIFT.png",
        "LOOKBACK_USABLE_FRACTION.png",
    ):
        (doc.parent / name).write_bytes((output / name).read_bytes())
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "_manifest.json"
    }
    manifest = {
        "schema_version": "courage_strict_continuous_march_drift_manifest_v1",
        "decision": "COMPLETE_MARCH_DRIFT_DIAGNOSTIC_CLOSURE",
        "source_experiment_id": contract["source_experiment_id"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "population_rows": {split: len(raw) for split, raw in raw_by_split.items()},
        "lookback_systematic_samples": {
            split: int(
                lookback.loc[lookback.split.eq(split), "sampled_sequences"].max()
            )
            for split in raw_by_split
        },
        "runtime_seconds": time.time() - started,
        "april_or_later_read": False,
        "patchtst_retrained": False,
        "checkpoint_reselected_or_promoted": False,
        "refit_strategy_backtest_trading_remote_push": False,
        "outputs": outputs,
    }
    atomic_json(output / "_manifest.json", manifest)
    return doc


if __name__ == "__main__":
    print(run())
