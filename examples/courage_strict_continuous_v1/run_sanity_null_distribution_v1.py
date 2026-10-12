"""Paired initialization/shuffled-label null distribution diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from examples.courage_strict_continuous_v1.run_sanity_checks_v1 import (
    _atomic_json,
    _datasets,
    _read_json,
    _seed,
    _shuffled_replacement,
    _ShuffledTargetDataset,
)
from qlib.contrib.model.courage_strict_continuous_v1 import (
    StratifiedTimeCellDistributedSampler,
    continuous_scheduler,
    effective_target_mask,
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
CONTRACT_PATH = (
    ROOT
    / "examples/courage_strict_continuous_v1/sanity_null_distribution_contract_v1.json"
)
METRICS = ("balanced_accuracy", "auc", "mcc", "rank_ic")


class NullDistributionError(RuntimeError):
    """Raised when the paired null diagnostic drifts from its contract."""


def _context() -> tuple[dict[str, Any], Any, Path]:
    contract = _read_json(CONTRACT_PATH)
    if (
        contract.get("schema_version")
        != "courage_strict_sanity_null_distribution_contract_v1"
        or contract.get("conditions")
        != ["INITIALIZATION_ONLY", "SHUFFLED_LABEL_500_STEPS"]
        or contract.get("boundaries", {}).get("april_or_later_read") is not False
    ):
        raise NullDistributionError("null-distribution contract drift")
    config = load_config(ROOT / contract["source_config"])
    validate_identity(config)
    origin = config.origins[contract["origin"]]
    if origin["valid"] != ["2026-01-05", "2026-02-02"]:
        raise NullDistributionError("fixed January Valid drift")
    output = ROOT / contract["outputs"]["artifact_root"]
    output.mkdir(parents=True, exist_ok=True)
    return contract, config, output


def _auc(truth: np.ndarray, score: np.ndarray) -> float:
    positive = int(truth.sum())
    negative = int(len(truth) - positive)
    if positive == 0 or negative == 0:
        return math.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
    return float(
        (ranks[truth].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def _direction_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = target > 0
    predicted = prediction > 0
    tp = int((truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    recall_up = tp / (tp + fn) if tp + fn else math.nan
    recall_down = tn / (tn + fp) if tn + fp else math.nan
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "balanced_accuracy": float((recall_up + recall_down) / 2),
        "auc": _auc(truth, prediction),
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator else 0.0,
        "target_up_rate": float(truth.mean()),
        "prediction_up_rate": float(predicted.mean()),
    }


def _daily_metrics(
    *,
    seed: int,
    condition: str,
    horizon: int,
    calendar: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    active = target != 0.0
    frame = pd.DataFrame(
        {
            "calendar_index": calendar[active].astype(np.int64),
            "target": target[active].astype(np.float64),
            "prediction": prediction[active].astype(np.float64),
        }
    )
    frame["trading_day_index"] = frame["calendar_index"] // 240
    timestamp_ic = (
        frame.groupby("calendar_index", sort=False)
        .apply(
            lambda value: value["prediction"].corr(
                value["target"], method="spearman"
            ),
            include_groups=False,
        )
        .rename("rank_ic")
        .reset_index()
    )
    timestamp_ic["trading_day_index"] = timestamp_ic["calendar_index"] // 240
    rank_by_day = timestamp_ic.groupby("trading_day_index").rank_ic.mean()
    rows: list[dict[str, Any]] = []
    for day, group in frame.groupby("trading_day_index", sort=True):
        metrics = _direction_metrics(
            group.target.to_numpy(), group.prediction.to_numpy()
        )
        rows.append(
            {
                "seed": seed,
                "condition": condition,
                "horizon": horizon,
                "trading_day_index": int(day),
                "rows": len(group),
                **metrics,
                "rank_ic": float(rank_by_day.loc[day]),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or result[list(METRICS)].isna().any().any():
        raise NullDistributionError("daily metric contains missing values")
    return result


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    scalers: dict[str, Any],
    seed: int,
    condition: str,
) -> pd.DataFrame:
    model.eval()
    values: dict[int, dict[str, list[np.ndarray]]] = {
        horizon: {"target": [], "prediction": [], "calendar": []}
        for horizon in HORIZONS_V1
    }
    for batch in loader:
        moved = _move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = _forward(model, moved).float().cpu().numpy()
        raw_target = batch["raw_targets"].numpy()
        mask = batch["target_mask"].numpy().astype(bool, copy=False)
        calendar = batch["calendar_index"].numpy()
        for head, horizon in enumerate(HORIZONS_V1):
            active = mask[:, head]
            stats = scalers["targets"][str(horizon)]
            raw_prediction = (
                prediction[:, head] * float(stats["iqr"])
                + float(stats["median"])
            )
            values[horizon]["target"].append(raw_target[active, head])
            values[horizon]["prediction"].append(raw_prediction[active])
            values[horizon]["calendar"].append(calendar[active])
    return pd.concat(
        [
            _daily_metrics(
                seed=seed,
                condition=condition,
                horizon=horizon,
                calendar=np.concatenate(item["calendar"]),
                target=np.concatenate(item["target"]),
                prediction=np.concatenate(item["prediction"]),
            )
            for horizon, item in values.items()
        ],
        ignore_index=True,
    )


def run_seed(seed: int) -> dict[str, Any]:
    contract, config, output = _context()
    if seed not in contract["seeds"]:
        raise NullDistributionError("unregistered seed")
    if not torch.cuda.is_available():
        raise NullDistributionError("paired null diagnostic requires CUDA")
    _seed(seed)
    device = torch.device("cuda", 0)
    raw, train, valid, scalers = _datasets(
        config,
        {
            "origin": contract["origin"],
            "scope": {
                "train": config.origins[contract["origin"]]["train"],
                "valid": config.origins[contract["origin"]]["valid"],
            },
        },
    )
    spec = contract["shuffled_training"]
    valid_loader = DataLoader(
        valid,
        batch_size=int(spec["valid_batch_size"]),
        shuffle=False,
        num_workers=int(spec["workers"]),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    vocabulary = _read_json(config.provider_root / "industry_vocabulary.json")
    model = PatchTSTCourageStrictV1(industry_vocab_size=len(vocabulary)).to(device)
    started = time.time()
    initial_daily = _evaluate(
        model,
        valid_loader,
        device=device,
        scalers=scalers,
        seed=seed,
        condition="INITIALIZATION_ONLY",
    )

    replacement, permutation_hashes = _shuffled_replacement(raw, scalers, seed)
    shuffled = _ShuffledTargetDataset(train, replacement)
    sampler = StratifiedTimeCellDistributedSampler(
        raw.ends, num_replicas=1, rank=0, seed=seed
    )
    train_loader = DataLoader(
        shuffled,
        batch_size=int(spec["batch_size"]),
        sampler=sampler,
        shuffle=False,
        num_workers=int(spec["workers"]),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    steps = int(spec["steps"])
    scheduler = continuous_scheduler(
        optimizer, total_steps=steps, warmup_fraction=0.05
    )
    model.train()
    curve: list[dict[str, float]] = []
    for step, batch in enumerate(train_loader, start=1):
        moved = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = _forward(model, moved)
            mask = effective_target_mask(moved)
            point = functional.huber_loss(
                prediction.float(), moved["targets"].float(), delta=1.0, reduction="none"
            )
            loss = (
                (point * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)
            ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            curve.append(
                {
                    "global_step": step,
                    "train_shuffled_standardized_huber": float(loss.detach().cpu()),
                }
            )
        if step >= steps:
            break
    del replacement
    shuffled_daily = _evaluate(
        model,
        valid_loader,
        device=device,
        scalers=scalers,
        seed=seed,
        condition="SHUFFLED_LABEL_500_STEPS",
    )
    daily = pd.concat([initial_daily, shuffled_daily], ignore_index=True)
    seed_root = output / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    daily.to_csv(seed_root / "daily_metrics.csv", index=False)
    _atomic_json(seed_root / "train_curve.json", curve)
    result = {
        "schema_version": "courage_strict_sanity_null_seed_result_v1",
        "decision": "PASS_PAIRED_NULL_SEED_COMPLETE",
        "seed": seed,
        "valid_rows": len(valid),
        "trading_days": int(daily.trading_day_index.nunique()),
        "daily_metric_rows": len(daily),
        "training_steps": steps,
        "permutation_sha256": permutation_hashes,
        "runtime_seconds": time.time() - started,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "april_or_later_read": False,
        "checkpoint_written_or_promoted": False,
    }
    _atomic_json(seed_root / "result.json", result)
    return result


def _circular_block_sample(
    values: np.ndarray, *, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    count = len(values)
    blocks = math.ceil(count / block_length)
    starts = rng.integers(0, count, size=blocks)
    indices = np.concatenate(
        [
            (np.arange(start, start + block_length, dtype=np.int64) % count)
            for start in starts
        ]
    )[:count]
    return values[indices]


def _bootstrap_ci(
    per_seed: dict[int, np.ndarray],
    *,
    block_length: int,
    replications: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    seeds = np.asarray(sorted(per_seed), dtype=np.int64)
    observed = float(np.mean([per_seed[int(seed)].mean() for seed in seeds]))
    draws = np.empty(replications, dtype=np.float64)
    for index in range(replications):
        selected = rng.choice(seeds, size=len(seeds), replace=True)
        draws[index] = np.mean(
            [
                _circular_block_sample(
                    per_seed[int(seed)], block_length=block_length, rng=rng
                ).mean()
                for seed in selected
            ]
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return observed, float(lower), float(upper)


def _single_seed_ci(
    values: np.ndarray,
    *,
    block_length: int,
    replications: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    draws = np.asarray(
        [
            _circular_block_sample(values, block_length=block_length, rng=rng).mean()
            for _ in range(replications)
        ]
    )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def aggregate() -> dict[str, Any]:
    contract, _, output = _context()
    seed_results = [
        _read_json(output / f"seed_{seed}" / "result.json")
        for seed in contract["seeds"]
    ]
    if any(item["decision"] != "PASS_PAIRED_NULL_SEED_COMPLETE" for item in seed_results):
        raise NullDistributionError("one or more paired seed runs are incomplete")
    daily = pd.concat(
        [
            pd.read_csv(output / f"seed_{seed}" / "daily_metrics.csv")
            for seed in contract["seeds"]
        ],
        ignore_index=True,
    )
    daily.to_csv(output / "all_daily_metrics.csv", index=False)
    bootstrap = contract["bootstrap"]
    block_length = int(bootstrap["block_length_days"])
    replications = int(bootstrap["replications"])
    master_rng = np.random.default_rng(int(bootstrap["seed"]))
    thresholds = contract["escalation_rule"]["metrics"]
    minimum_seeds = int(
        contract["escalation_rule"]["minimum_same_direction_seed_cis"]
    )
    summary: list[dict[str, Any]] = []
    escalations: list[dict[str, Any]] = []
    for horizon in HORIZONS_V1:
        selected = daily.loc[daily.horizon.eq(horizon)].copy()
        for metric in METRICS:
            null = float(thresholds[metric])
            initial = {
                int(seed): group.sort_values("trading_day_index")[metric].to_numpy()
                for seed, group in selected.loc[
                    selected.condition.eq("INITIALIZATION_ONLY")
                ].groupby("seed")
            }
            shuffled = {
                int(seed): group.sort_values("trading_day_index")[metric].to_numpy()
                for seed, group in selected.loc[
                    selected.condition.eq("SHUFFLED_LABEL_500_STEPS")
                ].groupby("seed")
            }
            paired = {
                seed: shuffled[seed] - initial[seed] for seed in sorted(initial)
            }
            initial_ci = _bootstrap_ci(
                initial,
                block_length=block_length,
                replications=replications,
                rng=master_rng,
            )
            shuffled_ci = _bootstrap_ci(
                shuffled,
                block_length=block_length,
                replications=replications,
                rng=master_rng,
            )
            paired_ci = _bootstrap_ci(
                paired,
                block_length=block_length,
                replications=replications,
                rng=master_rng,
            )
            positive = 0
            negative = 0
            for seed, values in shuffled.items():
                lower, upper = _single_seed_ci(
                    values,
                    block_length=block_length,
                    replications=replications,
                    rng=np.random.default_rng(int(bootstrap["seed"]) + seed + horizon),
                )
                positive += int(lower > null)
                negative += int(upper < null)
            shuffled_positive = shuffled_ci[1] > null
            shuffled_negative = shuffled_ci[2] < null
            paired_positive = paired_ci[1] > 0
            paired_negative = paired_ci[2] < 0
            escalate_positive = (
                positive >= minimum_seeds
                and shuffled_positive
                and paired_positive
            )
            escalate_negative = (
                negative >= minimum_seeds
                and shuffled_negative
                and paired_negative
            )
            escalate = escalate_positive or escalate_negative
            row = {
                "horizon": horizon,
                "metric": metric,
                "null": null,
                "initial_mean": initial_ci[0],
                "initial_ci_low": initial_ci[1],
                "initial_ci_high": initial_ci[2],
                "shuffled_mean": shuffled_ci[0],
                "shuffled_ci_low": shuffled_ci[1],
                "shuffled_ci_high": shuffled_ci[2],
                "paired_delta_mean": paired_ci[0],
                "paired_delta_ci_low": paired_ci[1],
                "paired_delta_ci_high": paired_ci[2],
                "positive_seed_cis": positive,
                "negative_seed_cis": negative,
                "escalate_leakage_investigation": escalate,
            }
            summary.append(row)
            if escalate:
                escalations.append(row)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(output / "bootstrap_summary.csv", index=False)
    decision = (
        "ESCALATE_TO_LEAKAGE_INVESTIGATION"
        if escalations
        else "PASS_NO_REPEATABLE_SHUFFLED_LABEL_LEAKAGE_PATTERN"
    )
    manifest = {
        "schema_version": "courage_strict_sanity_null_distribution_manifest_v1",
        "decision": decision,
        "seed_runs": seed_results,
        "bootstrap": bootstrap,
        "escalation_rule": contract["escalation_rule"],
        "escalations": escalations,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "april_or_later_read": False,
        "checkpoint_written_or_promoted": False,
    }
    _atomic_json(output / "_null_distribution_manifest.json", manifest)
    _render_report(contract, manifest, summary)
    return manifest


def _render_report(
    contract: dict[str, Any], manifest: dict[str, Any], summary: list[dict[str, Any]]
) -> None:
    lines = [
        "# Courage Strict Continuous V1 Shuffled-label Null Distribution",
        "",
        "## 结论",
        "",
        f"判定：`{manifest['decision']}`。10 个 seed 均完成同初始化配对的 initialization-only 与 500-step shuffled-label，并在完整 January Valid 上按交易日做 3 日 circular moving-block bootstrap。",
        "",
        "升级泄漏调查必须同时满足：shuffled 聚合 95% CI 排除随机值、shuffled−initialization 配对 CI 同方向排除 0、至少 4/10 seed 的单独日期块 CI 同方向排除随机值。",
        "",
        "| horizon | metric | init mean [95% CI] | shuffled mean [95% CI] | paired delta [95% CI] | +seed/-seed | escalation |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['horizon']} | {row['metric']} | "
            f"{row['initial_mean']:.5f} [{row['initial_ci_low']:.5f}, {row['initial_ci_high']:.5f}] | "
            f"{row['shuffled_mean']:.5f} [{row['shuffled_ci_low']:.5f}, {row['shuffled_ci_high']:.5f}] | "
            f"{row['paired_delta_mean']:.5f} [{row['paired_delta_ci_low']:.5f}, {row['paired_delta_ci_high']:.5f}] | "
            f"{row['positive_seed_cis']}/{row['negative_seed_cis']} | "
            f"{'YES' if row['escalate_leakage_investigation'] else 'NO'} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- initialization-only 控制随机网络本身对真实特征产生结构化排序的可能性。",
        "- 日期 block bootstrap 避免把高度重叠的分钟 Label 行数误当成独立样本量。",
        "- 本实验仅诊断是否存在可重复的 shuffled-label 异常；不证明正式模型有效，也不改变正式模型基线门失败。",
        "- 未读取四月或以后数据，未写入或晋升任何 checkpoint。",
        "",
    ]
    document = ROOT / contract["outputs"]["documentation"]
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seed", "aggregate"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "seed":
        if args.seed is None:
            raise NullDistributionError("--seed is required")
        result = run_seed(args.seed)
    else:
        result = aggregate()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
