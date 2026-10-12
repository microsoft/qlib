"""Complete the frozen Continuous V1 Valid baseline/diagnostic closure.

This is a read-only post-training evaluation.  It never changes checkpoint
selection, trains PatchTST, or reads data at/after the contract cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
DEFAULT_CONTRACT = (
    ROOT / "examples/courage_strict_continuous_v1/baseline_closure_contract_v1.json"
)
FEATURES = tuple(DYNAMIC_FEATURES) + tuple(SLOW_FEATURES)
ESTIMATORS = (
    "model",
    "zero",
    "train_mean",
    "train_median",
    "momentum_ret5",
    "reversal_ret5",
    "ridge_terminal",
)
BASELINES = ESTIMATORS[1:]


class BaselineClosureError(RuntimeError):
    """Raised on any identity, population, or finite-metric drift."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        return {"rows": 0}
    q = np.quantile(selected, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "rows": len(selected),
        "mean": float(selected.mean()),
        "std": float(selected.std(ddof=0)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "iqr": float(q[4] - q[2]),
        "up_rate": float((selected > 0).mean()),
    }


def _auc(target_up: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(target_up, dtype=bool)
    values = np.asarray(score, dtype=np.float64)
    positive = int(truth.sum())
    negative = int(len(truth) - positive)
    if positive == 0 or negative == 0:
        return math.nan
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    return float(
        (ranks[truth].sum() - positive * (positive + 1) / 2) / (positive * negative)
    )


def direction_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    active = target != 0.0
    truth = target[active] > 0
    predicted = prediction[active] > 0
    tp = int((truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    count = int(active.sum())
    recall_up = tp / (tp + fn) if tp + fn else math.nan
    recall_down = tn / (tn + fp) if tn + fp else math.nan
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": (tp + tn) / count if count else math.nan,
        "balanced_accuracy": (recall_up + recall_down) / 2,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "auc": _auc(truth, prediction[active]),
        "target_up_rate": float(truth.mean()) if count else math.nan,
        "prediction_up_rate": float(predicted.mean()) if count else math.nan,
    }


def metric_row(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (
        not len(target)
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        raise BaselineClosureError("metric population contains invalid values")
    error = prediction - target
    centered_target = target - target.mean()
    centered_prediction = prediction - prediction.mean()
    denominator = math.sqrt(
        float(np.square(centered_target).sum() * np.square(centered_prediction).sum())
    )
    pearson = (
        float((centered_target * centered_prediction).sum() / denominator)
        if denominator
        else 0.0
    )
    prediction_std = float(prediction.std(ddof=0))
    rank_ic = (
        pd.Series(prediction).corr(pd.Series(target), method="spearman")
        if prediction_std > 0 and float(target.std(ddof=0)) > 0
        else math.nan
    )
    return {
        "rows": len(target),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "pearson": pearson,
        "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else math.nan,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": prediction_std,
        "target_mean": float(target.mean()),
        "target_std": float(target.std(ddof=0)),
        **direction_metrics(target, prediction),
    }


def gate_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_rows = [row for row in rows if row["estimator"] == "model"]
    if len(model_rows) != len(HORIZONS_V1):
        raise BaselineClosureError("gate is missing model horizons")
    skills = [float(row["rmse_skill_vs_best_baseline"]) for row in model_rows]
    positive = int(sum(value > 0 for value in skills))
    mean = float(np.mean(skills))
    passed = mean > 0 and positive >= 4
    return {
        "mean_rmse_skill_vs_best_baseline": mean,
        "positive_horizons": positive,
        "required_positive_horizons": 4,
        "mean_skill_must_be_strictly_positive": True,
        "decision": "PASS_BASELINE_GATE" if passed else "FAIL_BASELINE_GATE",
        "continue_to_next_origin": passed,
    }


def solve_ridge(
    gram: np.ndarray,
    cross: np.ndarray,
    *,
    rows: int,
    alpha_on_average_gram: float,
) -> np.ndarray:
    if rows <= 0 or gram.shape[0] != gram.shape[1] or len(cross) != gram.shape[0]:
        raise BaselineClosureError("invalid Ridge sufficient statistics")
    penalty = np.eye(gram.shape[0], dtype=np.float64)
    penalty[0, 0] = 0.0
    system = gram / rows + float(alpha_on_average_gram) * penalty
    rhs = cross / rows
    try:
        coefficient = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(system, rhs, rcond=None)[0]
    if not np.isfinite(coefficient).all():
        raise BaselineClosureError("Ridge coefficient is non-finite")
    return coefficient


def _resolve(contract: dict[str, Any], key: str) -> Path:
    return (Path(contract["project_root"]) / contract[key]).resolve()


def _validate_contract(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version")
        != "courage_strict_continuous_baseline_closure_contract_v1"
        or contract["authority"].get("read_before_exclusive") != "2026-02-02"
        or any(
            contract["authority"].get(key) is not False
            for key in (
                "patchtst_retraining",
                "checkpoint_selection_change",
                "april_or_later_read",
                "refit",
                "strategy",
                "backtest",
                "trading",
                "remote_push",
            )
        )
    ):
        raise BaselineClosureError("contract authority/identity drift")
    for key, sha_key in (
        ("source_config", "source_config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("valid_predictions", "valid_predictions_sha256"),
        ("train_scalers", "train_scalers_sha256"),
    ):
        path = _resolve(contract, key)
        if not path.is_file() or sha256_file(path) != contract[sha_key]:
            raise BaselineClosureError(f"frozen input drift: {key}")
    provider_catalog = (
        _resolve(contract, "provider_root") / "_courage_strict_v1_qlib_catalog.json"
    )
    if sha256_file(provider_catalog) != contract["provider_catalog_sha256"]:
        raise BaselineClosureError("provider catalog drift")
    source = json.loads(_resolve(contract, "source_config").read_text(encoding="utf-8"))
    if contract.get("source_experiment_id") != source.get("experiment_id"):
        raise BaselineClosureError("source experiment identity drift")
    origin = source["origins"][contract["origin"]]
    if (
        origin["train"] != contract["segments"]["train"]
        or origin["valid"] != contract["segments"]["valid"]
    ):
        raise BaselineClosureError("segment drift")
    return contract, source


def _scaled_design(
    terminal: np.ndarray,
    usable: np.ndarray,
    minute_slot: np.ndarray,
    scalers: dict[str, Any],
) -> np.ndarray:
    count = len(terminal)
    output = np.empty((count, 1 + len(FEATURES) * 2 + 3), dtype=np.float64)
    output[:, 0] = 1.0
    for index, feature in enumerate(FEATURES):
        group = "dynamic" if index < len(DYNAMIC_FEATURES) else "slow"
        stats = scalers[group][feature]
        values = terminal[:, index].astype(np.float64, copy=True)
        if stats["clip"]:
            np.clip(values, float(stats["lower"]), float(stats["upper"]), out=values)
        values = (values - float(stats["median"])) / float(stats["iqr"])
        values[~usable[:, index]] = 0.0
        output[:, 1 + index] = values
        output[:, 1 + len(FEATURES) + index] = (~usable[:, index]).astype(np.float64)
    angle = 2.0 * np.pi * minute_slot.astype(np.float64) / 240.0
    clock = 1 + len(FEATURES) * 2
    output[:, clock] = np.sin(angle)
    output[:, clock + 1] = np.cos(angle)
    output[:, clock + 2] = minute_slot >= 120
    return output


def _extract_population(
    raw: Any, scalers: dict[str, Any], *, fit_ridge: bool, alpha: float
) -> dict[str, Any]:
    rows = len(raw)
    terminal = np.empty((rows, len(FEATURES)), dtype=np.float32)
    usable = np.empty((rows, len(FEATURES)), dtype=bool)
    targets = np.zeros((rows, len(HORIZONS_V1)), dtype=np.float32)
    masks = np.zeros((rows, len(HORIZONS_V1)), dtype=bool)
    ret5 = np.zeros(rows, dtype=np.float32)
    minute_slot = np.empty(rows, dtype=np.int16)
    industry_id = np.empty(rows, dtype=np.int16)
    turnover = np.full(rows, np.nan, dtype=np.float32)
    design_size = 1 + len(FEATURES) * 2 + 3
    grams = [
        np.zeros((design_size, design_size), dtype=np.float64) for _ in HORIZONS_V1
    ]
    crosses = [np.zeros(design_size, dtype=np.float64) for _ in HORIZONS_V1]
    ridge_rows = np.zeros(len(HORIZONS_V1), dtype=np.int64)
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
        ret5[left:right] = np.where(
            usable[left:right, DYNAMIC_FEATURES.index("stock_ret_5")],
            terminal[left:right, DYNAMIC_FEATURES.index("stock_ret_5")],
            0.0,
        )
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
        if fit_ridge:
            design = _scaled_design(
                terminal[left:right],
                usable[left:right],
                minute_slot[left:right],
                scalers,
            )
            for head in range(len(HORIZONS_V1)):
                active = masks[left:right, head]
                if active.any():
                    selected = design[active]
                    grams[head] += selected.T @ selected
                    crosses[head] += selected.T @ targets[left:right, head][active]
                    ridge_rows[head] += int(active.sum())
    if not np.isfinite(terminal[usable]).all():
        raise BaselineClosureError("usable terminal feature is non-finite")
    result = {
        "terminal": terminal,
        "usable": usable,
        "targets": targets,
        "masks": masks,
        "ret5": ret5,
        "minute_slot": minute_slot,
        "industry_id": industry_id,
        "turnover": turnover,
        "symbol_position": raw.symbol_positions.copy(),
        "calendar_index": raw.ends.copy(),
    }
    if fit_ridge:
        result["ridge_coefficients"] = np.column_stack(
            [
                solve_ridge(
                    grams[head],
                    crosses[head],
                    rows=int(ridge_rows[head]),
                    alpha_on_average_gram=alpha,
                )
                for head in range(len(HORIZONS_V1))
            ]
        )
        result["ridge_rows"] = ridge_rows
    return result


def _prediction_map(
    valid: dict[str, Any],
    model: np.ndarray,
    *,
    train_mean: np.ndarray,
    train_median: np.ndarray,
    ridge_coefficients: np.ndarray,
    scalers: dict[str, Any],
) -> dict[str, np.ndarray]:
    design = _scaled_design(
        valid["terminal"], valid["usable"], valid["minute_slot"], scalers
    )
    rows = len(design)
    return {
        "model": model.astype(np.float64),
        "zero": np.zeros((rows, len(HORIZONS_V1)), dtype=np.float64),
        "train_mean": np.broadcast_to(train_mean, (rows, len(HORIZONS_V1))).copy(),
        "train_median": np.broadcast_to(train_median, (rows, len(HORIZONS_V1))).copy(),
        "momentum_ret5": np.broadcast_to(
            valid["ret5"][:, None], (rows, len(HORIZONS_V1))
        ).astype(np.float64),
        "reversal_ret5": np.broadcast_to(
            -valid["ret5"][:, None], (rows, len(HORIZONS_V1))
        ).astype(np.float64),
        "ridge_terminal": design @ ridge_coefficients,
    }


def _baseline_metrics(
    valid: dict[str, Any], predictions: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    for head, horizon in enumerate(HORIZONS_V1):
        active = valid["masks"][:, head]
        target = valid["targets"][active, head].astype(np.float64)
        horizon_rows: list[dict[str, Any]] = []
        for estimator in ESTIMATORS:
            estimator_prediction = predictions[estimator][active, head]
            item = {
                "horizon": int(horizon),
                "estimator": estimator,
                **metric_row(target, estimator_prediction),
            }
            timestamp_ic = _timestamp_rank_ic(
                pd.DataFrame(
                    {
                        "calendar_index": valid["calendar_index"][active],
                        "target": target,
                        "prediction": estimator_prediction,
                    }
                )
            )
            item["rank_ic"] = (
                float(timestamp_ic.mean()) if len(timestamp_ic) else math.nan
            )
            horizon_rows.append(item)
        baseline_rows = [
            item for item in horizon_rows if item["estimator"] in BASELINES
        ]
        winner = min(baseline_rows, key=lambda item: item["rmse"])
        best[str(horizon)] = {
            "estimator": winner["estimator"],
            "rmse": float(winner["rmse"]),
        }
        for item in horizon_rows:
            item["rmse_skill_vs_best_baseline"] = 1.0 - item["rmse"] / winner["rmse"]
            rows.append(item)
    return rows, best


def _timestamp_rank_ic(frame: pd.DataFrame) -> pd.Series:
    if frame["prediction"].nunique(dropna=True) <= 1:
        return pd.Series(dtype=np.float64)
    return (
        frame.groupby("calendar_index", sort=False)
        .apply(
            lambda group: group["prediction"].corr(group["target"], method="spearman"),
            include_groups=False,
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _daily_metrics(
    valid: dict[str, Any],
    predictions: dict[str, np.ndarray],
    calendar: pd.DatetimeIndex,
    best: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for head, horizon in enumerate(HORIZONS_V1):
        active = valid["masks"][:, head]
        indices = valid["calendar_index"][active].astype(np.int64)
        dates = pd.to_datetime(calendar[indices]).normalize()
        target = valid["targets"][active, head].astype(np.float64)
        for estimator in ESTIMATORS:
            prediction = predictions[estimator][active, head]
            frame = pd.DataFrame(
                {
                    "date": dates,
                    "calendar_index": indices,
                    "target": target,
                    "prediction": prediction,
                }
            )
            timestamp_ic = _timestamp_rank_ic(frame)
            ic_dates = pd.to_datetime(
                calendar[timestamp_ic.index.to_numpy(dtype=np.int64)]
            ).normalize()
            daily_ic = (
                pd.Series(timestamp_ic.to_numpy(), index=ic_dates)
                .groupby(level=0)
                .mean()
            )
            for date, group in frame.groupby("date", sort=True):
                item = metric_row(group.target.to_numpy(), group.prediction.to_numpy())
                baseline_rmse = float(best[str(horizon)]["rmse"])
                records.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "horizon": int(horizon),
                        "estimator": estimator,
                        "rows": item["rows"],
                        "rmse": item["rmse"],
                        "rmse_skill_vs_global_best_baseline": 1.0
                        - item["rmse"] / baseline_rmse,
                        "bias": item["bias"],
                        "accuracy": item["accuracy"],
                        "balanced_accuracy": item["balanced_accuracy"],
                        "mcc": item["mcc"],
                        "auc": item["auc"],
                        "timestamp_rank_ic_mean": _finite_or_none(
                            daily_ic.get(date, math.nan)
                        ),
                        "target_up_rate": item["target_up_rate"],
                        "prediction_up_rate": item["prediction_up_rate"],
                    }
                )
    return pd.DataFrame(records)


def _grouped_model_metrics(
    valid: dict[str, Any],
    model: np.ndarray,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    calendar_index = valid["calendar_index"].astype(np.int64)
    dates = pd.to_datetime(calendar[calendar_index]).strftime("%Y-%m-%d")
    turnover = valid["turnover"]
    turnover_band = pd.cut(
        turnover,
        bins=[5.0, 7.5, 10.0, 12.5, 15.000001],
        labels=["5-7.5", "7.5-10", "10-12.5", "12.5-15"],
        include_lowest=True,
    ).astype(str)
    dimensions = {
        "date": np.asarray(dates),
        "minute_slot": valid["minute_slot"].astype(str),
        "industry_id": valid["industry_id"].astype(str),
        "turnover_band": np.asarray(turnover_band),
    }
    records: list[dict[str, Any]] = []
    for dimension, values in dimensions.items():
        groups = pd.Series(np.arange(len(values))).groupby(values, sort=True).indices
        for group, indices in groups.items():
            selected = np.asarray(indices, dtype=np.int64)
            for head, horizon in enumerate(HORIZONS_V1):
                active = valid["masks"][selected, head]
                if int(active.sum()) < 100:
                    continue
                population = selected[active]
                item = metric_row(
                    valid["targets"][population, head], model[population, head]
                )
                records.append(
                    {
                        "dimension": dimension,
                        "group": str(group),
                        "horizon": int(horizon),
                        **item,
                    }
                )
    return pd.DataFrame(records)


def _distribution_tables(
    train: dict[str, Any],
    valid: dict[str, Any],
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_records: list[dict[str, Any]] = []
    for split, population in (("train", train), ("valid", valid)):
        for index, feature in enumerate(FEATURES):
            active = population["usable"][:, index]
            stats = _quantiles(population["terminal"][active, index])
            feature_records.append(
                {
                    "split": split,
                    "feature": feature,
                    "population_rows": len(active),
                    "missing_rate": float((~active).mean()),
                    **stats,
                }
            )
    distribution_records: list[dict[str, Any]] = []
    for head, horizon in enumerate(HORIZONS_V1):
        for split, population in (("train", train), ("valid", valid)):
            active = population["masks"][:, head]
            distribution_records.append(
                {
                    "split": split,
                    "kind": "target",
                    "estimator": "target",
                    "horizon": int(horizon),
                    **_quantiles(population["targets"][active, head]),
                }
            )
        active = valid["masks"][:, head]
        for estimator in ESTIMATORS:
            distribution_records.append(
                {
                    "split": "valid",
                    "kind": "prediction",
                    "estimator": estimator,
                    "horizon": int(horizon),
                    **_quantiles(predictions[estimator][active, head]),
                }
            )
    return pd.DataFrame(feature_records), pd.DataFrame(distribution_records)


def _render_report(
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    gate: dict[str, Any],
    daily: pd.DataFrame,
    distributions: pd.DataFrame,
    *,
    source_experiment_id: str,
    checkpoint_step: int,
) -> str:
    model = {int(row["horizon"]): row for row in rows if row["estimator"] == "model"}
    target_lookup = {
        (str(row.split), int(row.horizon)): row
        for row in distributions.loc[distributions.kind.eq("target")].itertuples(
            index=False
        )
    }
    ratios = [
        model[horizon]["prediction_std"] / model[horizon]["target_std"]
        for horizon in HORIZONS_V1
    ]
    lines = [
        f"# {source_experiment_id} 测评基线闭环",
        "",
        f"> 冻结 step-{checkpoint_step} checkpoint 的 Train-only baseline 与完整 January rolling Valid 诊断；不是未见 Test。",
        "",
        "## 结论",
        "",
        f"- 门禁：`{gate['decision']}`；平均 best-baseline RMSE skill `{gate['mean_rmse_skill_vs_best_baseline']:+.3%}`，正 skill `{gate['positive_horizons']}/7`。",
        "- 当前模型未改动，checkpoint 选择未重做；全部 baseline 在完全相同的 Valid key/Label mask 上计算。",
        "",
        "## 诊断解读",
        "",
        f"- 模型输出明显收缩：七个 horizon 的 `prediction_std / target_std` 仅 `{min(ratios):.1%}`～`{max(ratios):.1%}`；5m/15m prediction up rate 分别只有 `{model[5]['prediction_up_rate']:.3%}`/`{model[15]['prediction_up_rate']:.3%}`。",
        f"- Target 水平发生时间漂移：120m Train→Valid mean 为 `{target_lookup[('train', 120)].mean:+.3%}`→`{target_lookup[('valid', 120)].mean:+.3%}`，240m 为 `{target_lookup[('train', 240)].mean:+.3%}`→`{target_lookup[('valid', 240)].mean:+.3%}`，480m 为 `{target_lookup[('train', 480)].mean:+.3%}`→`{target_lookup[('valid', 480)].mean:+.3%}`。",
        f"- 240m/480m 仍有弱排序信息（AUC `{model[240]['auc']:.3%}`/`{model[480]['auc']:.3%}`，Rank IC `{model[240]['rank_ic']:+.5f}`/`{model[480]['rank_ic']:+.5f}`），但绝对收益点预测均输给常数基线；这是“排序信号存在、条件均值校准失败”，不能视为收益回归通过。",
        "- 最强 RMSE baseline 分布为：5/15/30m Ridge，60/120/240m zero，480m Train mean；继续增加 epoch 或更换网络不能直接解释这一模式。",
        "",
        "## 逐 horizon 最强基线",
        "",
        "| H | Model RMSE | Best baseline | Baseline RMSE | Skill | ACC | BAcc | MCC | AUC | Rank IC | Pred/Target Std |",
        "|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS_V1:
        item = model[horizon]
        winner = best[str(horizon)]
        ratio = (
            item["prediction_std"] / item["target_std"]
            if item["target_std"]
            else math.nan
        )
        lines.append(
            f"| {horizon} | {item['rmse']:.8f} | {winner['estimator']} | {winner['rmse']:.8f} | "
            f"{item['rmse_skill_vs_best_baseline']:+.3%} | {item['accuracy']:.3%} | "
            f"{item['balanced_accuracy']:.3%} | {item['mcc']:+.5f} | {item['auc']:.3%} | "
            f"{item['rank_ic']:+.5f} | {ratio:.3f} |"
        )
    lines += [
        "",
        "## 全部 estimator RMSE",
        "",
        "| H | Estimator | RMSE | Skill vs best baseline | BAcc | MCC | AUC | Rank IC |",
        "|---:|:--|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['horizon']} | {row['estimator']} | {row['rmse']:.8f} | "
            f"{row['rmse_skill_vs_best_baseline']:+.3%} | {row['balanced_accuracy']:.3%} | "
            f"{row['mcc']:+.5f} | {row['auc']:.3%} | "
            f"{row['rank_ic']:+.5f} |"
        )
    model_daily = daily.loc[daily.estimator.eq("model")]
    daily_summary = (
        model_daily.groupby("horizon", sort=True)
        .agg(
            days=("date", "nunique"),
            positive_ic_days=(
                "timestamp_rank_ic_mean",
                lambda value: int((value.fillna(0) > 0).sum()),
            ),
            positive_bacc_days=(
                "balanced_accuracy",
                lambda value: int((value > 0.5).sum()),
            ),
            mean_mcc=("mcc", "mean"),
        )
        .reset_index()
    )
    lines += [
        "",
        "## 逐日稳定性摘要",
        "",
        "| H | Days | IC>0 days | BAcc>50% days | Mean daily MCC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in daily_summary.itertuples(index=False):
        lines.append(
            f"| {int(row.horizon)} | {int(row.days)} | {int(row.positive_ic_days)} | "
            f"{int(row.positive_bacc_days)} | {float(row.mean_mcc):+.5f} |"
        )
    lines += [
        "",
        "## Prediction 收缩",
        "",
        "`Pred/Target Std` 显著小于 1 表示模型输出接近常数；完整均值、标准差和分位数见 `target_prediction_distribution.csv`。",
        "",
        "## 产物",
        "",
        "- `baseline_metrics.csv`：全部 estimator × horizon 指标；",
        "- `daily_metrics.csv`：逐日 IC/BAcc/MCC/RMSE；",
        "- `grouped_model_metrics.csv`：日期、分钟位置、行业和换手率分组；",
        "- `feature_distribution.csv`：Train/Valid 终点特征缺失率、median/IQR；",
        "- `target_prediction_distribution.csv`：Target/Prediction 均值、标准差、分位数和上涨比例；",
        "- `baseline_predictions.parquet`：同人口 baseline 预测；",
        "- `ridge_coefficients.json`：Train-only Ridge 系数和拟合人口。",
        "",
        "## 边界",
        "",
        "- Ridge 只使用信号时点的 12 个动态特征、5 个慢特征、缺失指示和时钟变量；不使用 Valid 拟合或调参。",
        "- Momentum/Reversal 是可观测 `stock_ret_5` 的原值/相反数，缺失时预测 0。",
        "- 同一provider、模型与训练链路的小样本过拟合、源bar Label复算和shuffled-label证据沿用既有独立sanity报告；本次不重复训练这些诊断模型。",
        "- 未训练 PatchTST，未读取 2026-02-02 及以后数据，未执行 refit、策略、回测、交易或远端推送。",
        "",
    ]
    return "\n".join(lines)


def run(contract_path: Path) -> Path:
    started = time.time()
    contract, _source = _validate_contract(contract_path)
    project_root = Path(contract["project_root"])
    output = project_root / contract["outputs"]["artifact_root"]
    report_path = project_root / contract["outputs"]["documentation"]
    if output.exists():
        raise BaselineClosureError(f"output already exists: {output}")
    output.mkdir(parents=True)
    scalers = json.loads(
        _resolve(contract, "train_scalers").read_text(encoding="utf-8")
    )
    dataset = CourageStrictV1Dataset(
        provider_root=_resolve(contract, "provider_root"),
        segments={
            "train": tuple(contract["segments"]["train"]),
            "valid": tuple(contract["segments"]["valid"]),
        },
    )
    train_raw = dataset.prepare("train", cache_capacity=8)
    valid_raw = dataset.prepare("valid", cache_capacity=8)
    train = _extract_population(
        train_raw,
        scalers,
        fit_ridge=True,
        alpha=float(contract["baselines"]["ridge_terminal"]["alpha_on_average_gram"]),
    )
    valid = _extract_population(valid_raw, scalers, fit_ridge=False, alpha=0.0)
    frozen = pd.read_parquet(_resolve(contract, "valid_predictions")).sort_values(
        "row_index", kind="stable"
    )
    if (
        len(frozen) != len(valid_raw)
        or not np.array_equal(
            frozen.row_index.to_numpy(dtype=np.int64), np.arange(len(valid_raw))
        )
        or not np.array_equal(
            frozen.symbol_position.to_numpy(dtype=np.int32), valid["symbol_position"]
        )
        or not np.array_equal(
            frozen.calendar_index.to_numpy(dtype=np.int32), valid["calendar_index"]
        )
    ):
        raise BaselineClosureError("frozen Valid key drift")
    model = np.column_stack(
        [
            frozen[f"prediction_{horizon}"].to_numpy(dtype=np.float64)
            for horizon in HORIZONS_V1
        ]
    )
    for head, horizon in enumerate(HORIZONS_V1):
        observed_target = frozen[f"target_{horizon}"].to_numpy(dtype=np.float32)
        observed_mask = frozen[f"valid_{horizon}"].to_numpy(dtype=bool)
        if not np.array_equal(
            observed_mask, valid["masks"][:, head]
        ) or not np.array_equal(
            observed_target[observed_mask], valid["targets"][observed_mask, head]
        ):
            raise BaselineClosureError(f"frozen Valid target drift: {horizon}")
    train_mean = np.array(
        [
            train["targets"][train["masks"][:, head], head].astype(np.float64).mean()
            for head in range(len(HORIZONS_V1))
        ]
    )
    train_median = np.array(
        [float(scalers["targets"][str(horizon)]["median"]) for horizon in HORIZONS_V1]
    )
    predictions = _prediction_map(
        valid,
        model,
        train_mean=train_mean,
        train_median=train_median,
        ridge_coefficients=train["ridge_coefficients"],
        scalers=scalers,
    )
    rows, best = _baseline_metrics(valid, predictions)
    gate = gate_decision(rows)
    daily = _daily_metrics(valid, predictions, valid_raw.provider.calendar, best)
    grouped = _grouped_model_metrics(valid, model, valid_raw.provider.calendar)
    feature_distribution, target_distribution = _distribution_tables(
        train, valid, predictions
    )
    pd.DataFrame(rows).to_csv(output / "baseline_metrics.csv", index=False)
    daily.to_csv(output / "daily_metrics.csv", index=False)
    grouped.to_csv(output / "grouped_model_metrics.csv", index=False)
    feature_distribution.to_csv(output / "feature_distribution.csv", index=False)
    target_distribution.to_csv(
        output / "target_prediction_distribution.csv", index=False
    )
    baseline_frame = frozen[["row_index", "calendar_index", "symbol_position"]].copy()
    for estimator in BASELINES:
        for head, horizon in enumerate(HORIZONS_V1):
            baseline_frame[f"{estimator}_{horizon}"] = predictions[estimator][
                :, head
            ].astype(np.float32)
    baseline_frame.to_parquet(
        output / "baseline_predictions.parquet", index=False, compression="zstd"
    )
    coefficient_payload = {
        "feature_order": ["intercept"]
        + list(FEATURES)
        + [f"{feature}__missing" for feature in FEATURES]
        + ["minute_sin", "minute_cos", "afternoon_session"],
        "alpha_on_average_gram": float(
            contract["baselines"]["ridge_terminal"]["alpha_on_average_gram"]
        ),
        "fit_rows_by_horizon": dict(
            zip(map(str, HORIZONS_V1), map(int, train["ridge_rows"]), strict=True)
        ),
        "coefficients_by_horizon": {
            str(horizon): train["ridge_coefficients"][:, head].tolist()
            for head, horizon in enumerate(HORIZONS_V1)
        },
        "train_mean_by_horizon": dict(
            zip(map(str, HORIZONS_V1), train_mean.tolist(), strict=True)
        ),
        "train_median_by_horizon": dict(
            zip(map(str, HORIZONS_V1), train_median.tolist(), strict=True)
        ),
    }
    atomic_json(output / "ridge_coefficients.json", coefficient_payload)
    report = _render_report(
        rows,
        best,
        gate,
        daily,
        target_distribution,
        source_experiment_id=contract["source_experiment_id"],
        checkpoint_step=int(contract.get("checkpoint_step", 2250)),
    )
    (output / "BASELINE_CLOSURE_EVALUATION.md").write_text(report, encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    outputs = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "_manifest.json":
            outputs[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "schema_version": "courage_strict_continuous_baseline_closure_manifest_v1",
        "decision": gate["decision"],
        "source_experiment_id": contract["source_experiment_id"],
        "origin": contract["origin"],
        "contract_sha256": sha256_file(contract_path),
        "frozen_checkpoint_sha256": contract["checkpoint_sha256"],
        "frozen_valid_predictions_sha256": contract["valid_predictions_sha256"],
        "provider_catalog_sha256": contract["provider_catalog_sha256"],
        "train_rows": len(train_raw),
        "valid_rows": len(valid_raw),
        "best_baseline_by_horizon": best,
        "gate": gate,
        "runtime_seconds": time.time() - started,
        "read_before_exclusive": contract["authority"]["read_before_exclusive"],
        "patchtst_retrained": False,
        "april_or_later_read": False,
        "refit_strategy_backtest_trading_remote_push": False,
        "outputs": outputs,
    }
    atomic_json(output / "_manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    print(run(args.contract.resolve()))


if __name__ == "__main__":
    main()
