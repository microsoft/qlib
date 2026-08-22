"""Run bounded implementation sanity checks for continuous Courage Strict V1.

This diagnostic never mutates or promotes the frozen source checkpoint.  It
reads only the origin_2026_01 Train/Valid interval and the project-owned raw
minute snapshot used to build the accepted provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from qlib.contrib.data.courage_strict_v1.dataset import CourageStrictV1Dataset
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
    ROOT / "examples/courage_strict_continuous_v1/sanity_check_contract_v1.json"
)


class SanityCheckError(RuntimeError):
    """Raised when a diagnostic identity or invariant fails closed."""


class _TensorDictDataset(Dataset):
    """A small fixed sample set materialized once to remove repeated memmap I/O."""

    def __init__(self, items: list[dict[str, torch.Tensor]]) -> None:
        self.tensors = default_collate(items)

    def __len__(self) -> int:
        return int(next(iter(self.tensors.values())).shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SanityCheckError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _contract() -> tuple[dict[str, Any], Any, Path]:
    contract = _read_json(CONTRACT_PATH)
    if contract.get("schema_version") != "courage_strict_continuous_sanity_contract_v1":
        raise SanityCheckError("sanity contract schema drift")
    config_path = ROOT / contract["source_config"]
    config = load_config(config_path)
    validate_identity(config)
    origin = contract["origin"]
    source = config.origins[origin]
    if source["train"] != contract["scope"]["train"] or source["valid"] != contract["scope"]["valid"]:
        raise SanityCheckError("sanity segment drift")
    if contract["scope"]["april_or_later_read"] is not False:
        raise SanityCheckError("future-read boundary drift")
    output = ROOT / contract["outputs"]["artifact_root"]
    output.mkdir(parents=True, exist_ok=True)
    return contract, config, output


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scalers(config: Any, origin: str) -> dict[str, Any]:
    item = config.origins[origin]
    path = ROOT / item["accepted_scaler"]
    if sha256_file(path) != item["accepted_scaler_file_sha256"]:
        raise SanityCheckError("accepted Train scaler byte drift")
    return _read_json(path)


def _datasets(config: Any, contract: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    origin = contract["origin"]
    segment = config.origins[origin]
    facade = CourageStrictV1Dataset(
        provider_root=config.provider_root,
        segments={"train": tuple(segment["train"]), "valid": tuple(segment["valid"])},
    )
    scalers = _scalers(config, origin)
    raw_train = facade.prepare("train", cache_capacity=8)
    train = facade.prepare("train", scalers=scalers, cache_capacity=8)
    valid = facade.prepare("valid", scalers=scalers, cache_capacity=8)
    return raw_train, train, valid, scalers


def _active_mask_for_ends(raw: Any, position: int, ends: np.ndarray, horizon_index: int) -> np.ndarray:
    arrays = raw._arrays(position)
    horizon = HORIZONS_V1[horizon_index]
    active = (
        (arrays[f"label_valid_{horizon}"][ends] > 0.5)
        & np.isfinite(arrays[f"label_return_{horizon}"][ends])
        & (arrays[f"label_target_end_index_{horizon}"][ends] < raw.label_cutoff_index)
        & (arrays[f"label_available_index_{horizon}"][ends] < raw.label_cutoff_index)
    )
    slots = arrays["minute_slot"][ends].astype(np.int64)
    sessions = arrays["session_id"][ends].astype(np.int64)
    phase = np.where(sessions == 0, slots, slots - 121)
    if horizon_index == 4:
        active &= phase % 15 == 0
    elif horizon_index >= 5:
        active &= phase % 30 == 0
    return active


def _all_head_eligible(raw: Any) -> np.ndarray:
    eligible_parts: list[np.ndarray] = []
    for position in np.unique(raw.symbol_positions):
        indices = np.flatnonzero(raw.symbol_positions == position)
        ends = raw.ends[indices]
        active = np.ones(len(indices), dtype=bool)
        for head in range(len(HORIZONS_V1)):
            active &= _active_mask_for_ends(raw, int(position), ends, head)
        eligible_parts.append(indices[active])
    result = np.concatenate(eligible_parts)
    if len(result) < 512:
        raise SanityCheckError("not enough all-head mature samples")
    return result


def _time_stratified_choice(raw: Any, indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    ordered = indices[np.argsort(raw.ends[indices], kind="stable")]
    chunks = np.array_split(ordered, count)
    rng = np.random.default_rng(seed)
    selected = np.asarray([chunk[rng.integers(0, len(chunk))] for chunk in chunks], dtype=np.int64)
    if len(np.unique(selected)) != count:
        raise SanityCheckError("stratified selection duplicated a row")
    return selected


def _raw_record_map() -> dict[str, dict[str, Any]]:
    manifest = _read_json(
        ROOT / "data/courage_strict_v1/source/minute_snapshot/_source_identity_manifest.json"
    )
    if (
        manifest.get("schema_version")
        != "courage_strict_v1_minute_source_identity_v1"
        or manifest.get("decision") != "PASS_V1_MINUTE_SOURCE_CONSTRUCTED"
    ):
        raise SanityCheckError("raw minute snapshot is not accepted")
    return {
        Path(item["relative_path"]).stem: item
        for item in manifest["files"]
        if item.get("roles") == ["one_minute_OHLCVA_bars"]
    }


def run_label_recomputation() -> dict[str, Any]:
    contract, config, output = _contract()
    raw, _, _, _ = _datasets(config, contract)
    spec = contract["label_recomputation"]
    per_head = int(spec["pairs_per_horizon"])
    records = _raw_record_map()
    axis = pd.read_parquet(
        ROOT / "data/courage_strict_v1/source/official_axis/official_minute_axis.parquet"
    )
    axis = axis.loc[axis["exchange"].eq("SSE")].copy()
    axis = axis.sort_values(["session_date", "minute_slot"], kind="mergesort").reset_index(drop=True)
    ready = (
        pd.to_datetime(axis["feature_ready_time"], utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    provider_calendar = raw.provider.calendar.tz_localize(None)
    axis = axis.loc[ready.isin(provider_calendar)].reset_index(drop=True)
    ready = (
        pd.to_datetime(axis["feature_ready_time"], utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    if not np.array_equal(ready.to_numpy(), provider_calendar.to_numpy()):
        raise SanityCheckError("official-axis/provider calendar drift")
    candidates: list[dict[str, Any]] = []
    for head, horizon in enumerate(HORIZONS_V1):
        parts: list[np.ndarray] = []
        for position in np.unique(raw.symbol_positions):
            indices = np.flatnonzero(raw.symbol_positions == position)
            active = _active_mask_for_ends(raw, int(position), raw.ends[indices], head)
            if active.any():
                parts.append(indices[active])
        pool = np.concatenate(parts)
        chosen = _time_stratified_choice(raw, pool, per_head, int(spec["seed"]) + horizon)
        for index in chosen:
            position = int(raw.symbol_positions[index])
            end = int(raw.ends[index])
            candidates.append(
                {
                    "dataset_index": int(index),
                    "symbol_position": position,
                    "instrument": raw.provider.symbols[position],
                    "signal_time": str(provider_calendar[end]),
                    "entry_index": end,
                    "exit_index": end + horizon,
                    "horizon": horizon,
                    "provider_label": float(raw.provider.array(position, f"label_return_{horizon}")[end]),
                }
            )

    audited: list[dict[str, Any]] = []
    label_manifests = list(
        (ROOT / "artifacts/courage_strict_v1/stages/labels").glob(
            "authorization-*/_c3_label_manifest.json"
        )
    )
    if len(label_manifests) != 1:
        raise SanityCheckError("accepted C3 label store identity is ambiguous")
    label_manifest = _read_json(label_manifests[0])
    if label_manifest.get("decision") != "PASS_C3_B_LABEL_AND_SAMPLE_CATALOG":
        raise SanityCheckError("accepted C3 label store is not PASS")
    label_root = label_manifests[0].parent
    for symbol, group in pd.DataFrame(candidates).groupby("instrument", sort=True):
        source = records.get(symbol)
        if source is None:
            raise SanityCheckError(f"raw source identity missing: {symbol}")
        source_path = Path(source["absolute_path"])
        if sha256_file(source_path) != source["sha256"]:
            raise SanityCheckError(f"raw source byte drift: {symbol}")
        bars = pq.read_table(source_path, columns=["trade_time", "vol", "amount"]).to_pandas()
        if "trade_time" not in bars.columns:
            bars = bars.reset_index()
        bars["trade_time"] = pd.to_datetime(bars["trade_time"])
        if bars["trade_time"].duplicated().any():
            raise SanityCheckError(f"duplicate raw timestamp: {symbol}")
        bars = bars.set_index("trade_time")
        sample_path = label_root / "samples" / f"{symbol}.parquet"
        sample_record = next(
            (
                item
                for item in label_manifest["files"]
                if item["path"] == f"samples/{symbol}.parquet"
            ),
            None,
        )
        if sample_record is None or sha256_file(sample_path) != sample_record["sha256"]:
            raise SanityCheckError(f"accepted sample time identity drift: {symbol}")
        sample_columns = ["signal_time", "entry_window_end"] + [
            f"target_window_end_vwap1_{horizon}m" for horizon in HORIZONS_V1
        ]
        samples = pq.read_table(sample_path, columns=sample_columns).to_pandas()
        samples["signal_time_key"] = (
            pd.to_datetime(samples["signal_time"], utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
        if samples["signal_time_key"].duplicated().any():
            raise SanityCheckError(f"duplicate accepted sample time: {symbol}")
        samples = samples.set_index("signal_time_key")
        for item in group.to_dict("records"):
            signal_time = pd.Timestamp(item["signal_time"])
            try:
                sample = samples.loc[signal_time]
            except KeyError as exc:
                raise SanityCheckError(
                    f"provider signal absent from accepted sample store: {symbol}/{signal_time}"
                ) from exc
            entry_time = pd.Timestamp(sample["entry_window_end"]).tz_localize(None)
            exit_time = pd.Timestamp(
                sample[f"target_window_end_vwap1_{int(item['horizon'])}m"]
            ).tz_localize(None)
            try:
                entry = bars.loc[entry_time]
                exit_ = bars.loc[exit_time]
            except KeyError as exc:
                raise SanityCheckError(f"raw audit timestamp absent: {symbol}/{exc}") from exc
            if float(entry["vol"]) <= 0 or float(exit_["vol"]) <= 0:
                raise SanityCheckError(
                    "provider-valid audit row has non-positive raw volume: "
                    f"{symbol}/{item['horizon']}/{entry_time}/{exit_time}/"
                    f"{float(entry['vol'])}/{float(exit_['vol'])}"
                )
            entry_vwap = float(entry["amount"]) / float(entry["vol"])
            exit_vwap = float(exit_["amount"]) / float(exit_["vol"])
            recomputed64 = exit_vwap / entry_vwap - 1.0
            recomputed32 = np.float32(recomputed64)
            provider32 = np.float32(item["provider_label"])
            bit_exact = bool(recomputed32.view(np.uint32) == provider32.view(np.uint32))
            audited.append(
                item
                | {
                    "entry_bar_time": str(entry_time),
                    "exit_bar_time": str(exit_time),
                    "provider_signal_to_entry_bar_minutes": float(
                        (entry_time - signal_time).total_seconds() / 60.0
                    ),
                    "entry_vwap": entry_vwap,
                    "exit_vwap": exit_vwap,
                    "recomputed_float64": recomputed64,
                    "recomputed_float32": float(recomputed32),
                    "absolute_error_after_float32": float(abs(recomputed32 - provider32)),
                    "float32_bit_exact": bit_exact,
                    "source_sha256": source["sha256"],
                }
            )
    frame = pd.DataFrame(audited).sort_values(["horizon", "entry_index", "instrument"])
    frame.to_csv(output / "label_recomputation_rows.csv", index=False)
    passed = bool(frame["float32_bit_exact"].all() and len(frame) == per_head * len(HORIZONS_V1))
    result = {
        "schema_version": "courage_strict_label_recomputation_result_v1",
        "decision": "PASS_LABEL_RECOMPUTATION" if passed else "FAIL_LABEL_RECOMPUTATION",
        "rows": len(frame),
        "rows_per_horizon": {str(k): int(v) for k, v in frame.groupby("horizon").size().items()},
        "bit_exact_rows": int(frame["float32_bit_exact"].sum()),
        "maximum_absolute_error_after_float32": float(frame["absolute_error_after_float32"].max()),
        "ransac_used": False,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "april_or_later_read": False,
    }
    _atomic_json(output / "label_recomputation_result.json", result)
    return result


def _disable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0


@torch.no_grad()
def _memorization_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[float], float]:
    model.eval()
    numerator = torch.zeros(len(HORIZONS_V1), dtype=torch.float64, device=device)
    denominator = torch.zeros_like(numerator)
    for batch in loader:
        moved = _move(batch, device)
        prediction = _forward(model, moved)
        loss = functional.huber_loss(prediction.float(), moved["targets"].float(), delta=1.0, reduction="none")
        mask = effective_target_mask(moved)
        numerator += (loss * mask).sum(dim=0).double()
        denominator += mask.sum(dim=0).double()
    values = (numerator / denominator).cpu().tolist()
    return [float(v) for v in values], float(np.mean(values))


def run_small_sample_overfit() -> dict[str, Any]:
    contract, config, output = _contract()
    spec = contract["small_sample_overfit"]
    seed = int(spec["seed"])
    _seed(seed)
    if not torch.cuda.is_available():
        raise SanityCheckError("small-sample overfit requires CUDA")
    device = torch.device("cuda", 0)
    raw, train, _, _ = _datasets(config, contract)
    eligible = _all_head_eligible(raw)
    selected = _time_stratified_choice(raw, eligible, int(spec["samples"]), seed)
    np.save(output / "small_overfit_sample_indices.npy", selected, allow_pickle=False)
    subset = _TensorDictDataset([train[int(index)] for index in selected])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(subset, batch_size=int(spec["batch_size"]), shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(subset, batch_size=int(spec["batch_size"]), shuffle=False, num_workers=0, pin_memory=True)
    vocabulary = _read_json(config.provider_root / "industry_vocabulary.json")
    model = PatchTSTCourageStrictV1(industry_vocab_size=len(vocabulary)).to(device)
    _disable_dropout(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=0.0)
    maximum_steps = int(spec["maximum_steps"])
    interval = int(spec["evaluation_interval_steps"])
    history: list[dict[str, Any]] = []
    initial_head, initial = _memorization_loss(model, eval_loader, device)
    history.append({"global_step": 0, "mean_standardized_huber": initial, "per_head_standardized_huber": dict(zip(map(str, HORIZONS_V1), initial_head, strict=True))})
    step = 0
    started = time.time()
    while step < maximum_steps:
        model.train()
        for batch in train_loader:
            moved = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, moved)
            mask = effective_target_mask(moved)
            point = functional.huber_loss(prediction.float(), moved["targets"].float(), delta=1.0, reduction="none")
            denominator = mask.sum(dim=0).clamp_min(1)
            loss = ((point * mask).sum(dim=0) / denominator).mean()
            loss.backward()
            optimizer.step()
            step += 1
            if step % interval == 0 or step == maximum_steps:
                per_head, mean = _memorization_loss(model, eval_loader, device)
                history.append({"global_step": step, "mean_standardized_huber": mean, "per_head_standardized_huber": dict(zip(map(str, HORIZONS_V1), per_head, strict=True))})
                if mean <= float(spec["pass_mean_standardized_huber_lte"]) and max(per_head) <= float(spec["pass_each_head_standardized_huber_lte"]):
                    step = maximum_steps
                    break
                model.train()
            if step >= maximum_steps:
                break
    final = history[-1]
    per_head = list(final["per_head_standardized_huber"].values())
    passed = final["mean_standardized_huber"] <= float(spec["pass_mean_standardized_huber_lte"]) and max(per_head) <= float(spec["pass_each_head_standardized_huber_lte"])
    _atomic_json(output / "small_overfit_curve.json", history)
    figure, axis_plot = plt.subplots(figsize=(9, 5))
    axis_plot.plot([r["global_step"] for r in history], [r["mean_standardized_huber"] for r in history], marker="o", markersize=2)
    axis_plot.axhline(float(spec["pass_mean_standardized_huber_lte"]), color="tab:red", linestyle="--", label="pass threshold")
    axis_plot.set_yscale("log")
    axis_plot.set_xlabel("optimizer step")
    axis_plot.set_ylabel("fixed-512 standardized Huber")
    axis_plot.set_title("Small-sample memorization sanity check")
    axis_plot.grid(alpha=0.25)
    axis_plot.legend()
    figure.tight_layout()
    figure.savefig(output / "SMALL_SAMPLE_OVERFIT_CURVE.png", dpi=180)
    plt.close(figure)
    result = {
        "schema_version": "courage_strict_small_sample_overfit_result_v1",
        "decision": "PASS_SMALL_SAMPLE_OVERFIT" if passed else "FAIL_SMALL_SAMPLE_OVERFIT",
        "samples": len(selected),
        "steps_executed": int(history[-1]["global_step"]),
        "initial_mean_standardized_huber": initial,
        "final_mean_standardized_huber": float(final["mean_standardized_huber"]),
        "loss_reduction_fraction": float(1.0 - final["mean_standardized_huber"] / initial),
        "final_per_head_standardized_huber": final["per_head_standardized_huber"],
        "dropout": 0.0,
        "weight_decay": 0.0,
        "runtime_seconds": time.time() - started,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "april_or_later_read": False,
    }
    _atomic_json(output / "small_overfit_result.json", result)
    return result


class _ShuffledTargetDataset(Dataset):
    def __init__(self, base: Any, replacement: np.ndarray) -> None:
        self.base = base
        self.raw = base.raw
        self.replacement = replacement

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.base[index]
        targets = item["targets"].clone()
        replacement = self.replacement[index]
        usable = np.isfinite(replacement)
        if usable.any():
            targets[torch.from_numpy(usable)] = torch.from_numpy(replacement[usable])
        return item | {"targets": targets}


def _shuffled_replacement(raw: Any, scalers: dict[str, Any], seed: int) -> tuple[np.ndarray, dict[str, str]]:
    replacement = np.full((len(raw), len(HORIZONS_V1)), np.nan, dtype=np.float32)
    hashes: dict[str, str] = {}
    for head, horizon in enumerate(HORIZONS_V1):
        indices_parts: list[np.ndarray] = []
        values_parts: list[np.ndarray] = []
        for position in np.unique(raw.symbol_positions):
            indices = np.flatnonzero(raw.symbol_positions == position)
            ends = raw.ends[indices]
            active = _active_mask_for_ends(raw, int(position), ends, head)
            if active.any():
                arrays = raw._arrays(int(position))
                indices_parts.append(indices[active])
                values_parts.append(np.asarray(arrays[f"label_return_{horizon}"][ends[active]], dtype=np.float32))
        indices = np.concatenate(indices_parts)
        values = np.concatenate(values_parts).astype(np.float64)
        stats = scalers["targets"][str(horizon)]
        values = ((values - float(stats["median"])) / float(stats["iqr"])).astype(np.float32)
        rng = np.random.default_rng(seed + horizon)
        permutation = rng.permutation(len(values))
        shuffled = values[permutation]
        if np.array_equal(values.view(np.uint32), shuffled.view(np.uint32)):
            raise SanityCheckError("label permutation is identity")
        replacement[indices, head] = shuffled
        hashes[str(horizon)] = hashlib.sha256(indices.tobytes() + shuffled.tobytes()).hexdigest()
    return replacement, hashes


def _auc(truth: np.ndarray, score: np.ndarray) -> float:
    positive = int(truth.sum())
    negative = int(len(truth) - positive)
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
    return float((ranks[truth].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _random_control_metrics(calendar_index: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    active = target != 0.0
    target = target[active]
    prediction = prediction[active]
    calendar_index = calendar_index[active]
    truth = target > 0
    predicted = prediction > 0
    tp = int((truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    fp = int((~truth & predicted).sum())
    recall_up = tp / (tp + fn)
    recall_down = tn / (tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    frame = pd.DataFrame({"calendar_index": calendar_index, "target": target, "prediction": prediction})
    rank_ic = frame.groupby("calendar_index", sort=False).apply(lambda x: x["prediction"].corr(x["target"], method="spearman"), include_groups=False).dropna()
    return {
        "rows": len(target),
        "balanced_accuracy": float((recall_up + recall_down) / 2),
        "auc": _auc(truth, prediction),
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator else 0.0,
        "rank_ic_mean": float(rank_ic.mean()),
        "target_up_rate": float(truth.mean()),
        "prediction_up_rate": float(predicted.mean()),
    }


def run_shuffled_label(seed: int) -> dict[str, Any]:
    contract, config, output = _contract()
    spec = contract["shuffled_label"]
    if seed not in spec["seeds"]:
        raise SanityCheckError("unregistered shuffled-label seed")
    if not torch.cuda.is_available():
        raise SanityCheckError("shuffled-label control requires CUDA")
    _seed(seed)
    device = torch.device("cuda", 0)
    raw, train, valid, scalers = _datasets(config, contract)
    replacement, hashes = _shuffled_replacement(raw, scalers, seed)
    shuffled = _ShuffledTargetDataset(train, replacement)
    sampler = StratifiedTimeCellDistributedSampler(raw.ends, num_replicas=1, rank=0, seed=seed)
    train_loader = DataLoader(shuffled, batch_size=int(spec["batch_size"]), sampler=sampler, shuffle=False, num_workers=int(spec["workers"]), pin_memory=True, persistent_workers=True, prefetch_factor=4)
    valid_loader = DataLoader(valid, batch_size=int(spec["valid_batch_size"]), shuffle=False, num_workers=int(spec["workers"]), pin_memory=True, persistent_workers=True, prefetch_factor=4)
    vocabulary = _read_json(config.provider_root / "industry_vocabulary.json")
    model = PatchTSTCourageStrictV1(industry_vocab_size=len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    steps = int(spec["training_steps"])
    scheduler = continuous_scheduler(optimizer, total_steps=steps, warmup_fraction=0.05)
    started = time.time()
    model.train()
    train_curve: list[dict[str, float]] = []
    for step, batch in enumerate(train_loader, start=1):
        moved = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = _forward(model, moved)
            mask = effective_target_mask(moved)
            point = functional.huber_loss(prediction.float(), moved["targets"].float(), delta=1.0, reduction="none")
            loss = ((point * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            train_curve.append({"global_step": step, "train_shuffled_standardized_huber": float(loss.detach().cpu())})
        if step >= steps:
            break
    del replacement
    model.eval()
    collected = {h: {"target": [], "prediction": [], "calendar": []} for h in HORIZONS_V1}
    with torch.no_grad():
        for batch in valid_loader:
            moved = _move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = _forward(model, moved).float().cpu().numpy()
            raw_target = batch["raw_targets"].numpy()
            mask = batch["target_mask"].numpy().astype(bool, copy=False)
            calendar = batch["calendar_index"].numpy()
            for head, horizon in enumerate(HORIZONS_V1):
                active = mask[:, head]
                stats = scalers["targets"][str(horizon)]
                raw_prediction = prediction[:, head] * float(stats["iqr"]) + float(stats["median"])
                collected[horizon]["target"].append(raw_target[active, head].astype(np.float32))
                collected[horizon]["prediction"].append(raw_prediction[active].astype(np.float32))
                collected[horizon]["calendar"].append(calendar[active].astype(np.int32))
    bounds = spec["per_seed_random_bounds"]
    by_horizon: dict[str, Any] = {}
    pass_flags: list[bool] = []
    for horizon in HORIZONS_V1:
        item = collected[horizon]
        metrics = _random_control_metrics(np.concatenate(item["calendar"]), np.concatenate(item["target"]), np.concatenate(item["prediction"]))
        passed = (
            abs(metrics["balanced_accuracy"] - 0.5) <= float(bounds["balanced_accuracy_abs_from_half_lte"])
            and abs(metrics["auc"] - 0.5) <= float(bounds["auc_abs_from_half_lte"])
            and abs(metrics["mcc"]) <= float(bounds["mcc_abs_lte"])
            and abs(metrics["rank_ic_mean"]) <= float(bounds["rank_ic_abs_lte"])
        )
        by_horizon[str(horizon)] = metrics | {"within_random_bounds": passed}
        pass_flags.append(passed)
    seed_root = output / "shuffled_label" / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(seed_root / "train_curve.json", train_curve)
    result = {
        "schema_version": "courage_strict_shuffled_label_seed_result_v1",
        "decision": "PASS_SHUFFLED_LABEL_RANDOM_CONTROL" if all(pass_flags) else "FAIL_SHUFFLED_LABEL_RANDOM_CONTROL",
        "seed": seed,
        "training_steps": steps,
        "train_label_permutation_sha256": hashes,
        "full_fixed_valid": True,
        "valid_rows": len(valid),
        "by_horizon": by_horizon,
        "runtime_seconds": time.time() - started,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "april_or_later_read": False,
    }
    _atomic_json(seed_root / "result.json", result)
    return result


def aggregate() -> dict[str, Any]:
    contract, _, output = _contract()
    label = _read_json(output / "label_recomputation_result.json")
    overfit = _read_json(output / "small_overfit_result.json")
    shuffled = [_read_json(output / "shuffled_label" / f"seed_{seed}" / "result.json") for seed in contract["shuffled_label"]["seeds"]]
    decisions = [label["decision"], overfit["decision"]] + [item["decision"] for item in shuffled]
    passed = all(value.startswith("PASS_") for value in decisions)
    rows: list[dict[str, Any]] = []
    for item in shuffled:
        for horizon, metrics in item["by_horizon"].items():
            rows.append({"seed": item["seed"], "horizon": int(horizon)} | metrics)
    row_frame = pd.DataFrame(rows).sort_values(["seed", "horizon"])
    row_frame.to_csv(output / "shuffled_label_metrics.csv", index=False)
    failed_cells = row_frame.loc[~row_frame["within_random_bounds"]]
    failed_counts = failed_cells.groupby("horizon").size()
    seed_means = (
        row_frame.groupby("horizon", sort=True)[
            ["balanced_accuracy", "auc", "mcc", "rank_ic_mean"]
        ]
        .mean()
        .reset_index()
        .to_dict("records")
    )
    posthoc = {
        "diagnostic_only_not_a_gate_change": True,
        "cells_within_random_bounds": int(row_frame["within_random_bounds"].sum()),
        "total_cells": len(row_frame),
        "failed_cells": failed_cells[
            ["seed", "horizon", "balanced_accuracy", "auc", "mcc", "rank_ic_mean"]
        ].to_dict("records"),
        "horizons_failing_in_two_or_more_seeds": [
            int(horizon) for horizon, count in failed_counts.items() if count >= 2
        ],
        "three_seed_means": seed_means,
        "interpretation": (
            "NO_REPEATABLE_HORIZON_FAILURE_BUT_CONSERVATIVE_PER_SEED_GATE_FAILED"
            if not any(count >= 2 for count in failed_counts)
            else "REPEATABLE_HORIZON_FAILURE_REQUIRES_LEAKAGE_INVESTIGATION"
        ),
    }
    manifest = {
        "schema_version": "courage_strict_continuous_sanity_manifest_v1",
        "decision": "PASS_ALL_THREE_SANITY_CHECKS" if passed else "FAIL_ONE_OR_MORE_SANITY_CHECKS",
        "subdecisions": decisions,
        "label_recomputation": label,
        "small_sample_overfit": overfit,
        "shuffled_label": shuffled,
        "shuffled_label_posthoc_seed_consistency": posthoc,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_checkpoint_mutated": False,
        "checkpoint_promoted": False,
        "april_or_later_read": False,
        "strategy_or_backtest_executed": False,
    }
    _atomic_json(output / "_sanity_manifest.json", manifest)
    _render_report(contract, manifest, rows)
    return manifest


def _render_report(contract: dict[str, Any], manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    label = manifest["label_recomputation"]
    overfit = manifest["small_sample_overfit"]
    posthoc = manifest["shuffled_label_posthoc_seed_consistency"]
    lines = [
        "# Courage Strict Continuous V1 三项 Sanity Check 报告",
        "",
        "## 结论",
        "",
        f"总判定：`{manifest['decision']}`。本实验只读取 Origin-1 Train/January Valid 与其原始分钟源，未读取四月及以后数据，也未晋升 checkpoint。",
        "",
        "| 检查 | 判定 | 核心结果 |",
        "|---|---|---|",
        f"| 原始 Bar Label 独立复算 | `{label['decision']}` | {label['bit_exact_rows']}/{label['rows']} 条 float32 bit-exact；最大误差 {label['maximum_absolute_error_after_float32']:.3g} |",
        f"| 固定 512 条小样本过拟合 | `{overfit['decision']}` | Huber {overfit['initial_mean_standardized_huber']:.6g} → {overfit['final_mean_standardized_huber']:.6g}，下降 {overfit['loss_reduction_fraction']:.2%} |",
        f"| 3-seed shuffled-label | `{'PASS' if all(x['decision'].startswith('PASS_') for x in manifest['shuffled_label']) else 'FAIL'}` | 完整固定 Valid；每 seed × 7 heads 均按随机界限审计 |",
        "",
        "## 1. Label 独立复算",
        "",
        f"按每个 horizon {contract['label_recomputation']['pairs_per_horizon']} 条、合计 {label['rows']} 条进行时间分层抽取。直接读取不可变分钟 Parquet，以 `amount/volume` 分别计算 entry/exit VWAP，再计算收益并转为 float32 与 provider bin 比较。RANSAC 未使用，因为这是确定性数据核对，不是鲁棒拟合问题。",
        "",
        "逐条证据：`artifacts/courage_strict_continuous_v1/sanity_checks_v1/label_recomputation_rows.csv`。",
        "",
        "## 2. 小样本过拟合",
        "",
        f"固定 {overfit['samples']} 条七头均成熟且符合训练密度的样本；dropout=0、weight_decay=0，最多 {contract['small_sample_overfit']['maximum_steps']} steps。最终每头 Huber：",
        "",
        "| horizon | standardized Huber |",
        "|---:|---:|",
    ]
    for horizon, value in overfit["final_per_head_standardized_huber"].items():
        lines.append(f"| {horizon} | {value:.8g} |")
    lines += [
        "",
        "曲线：![Small-sample overfit curve](../../artifacts/courage_strict_continuous_v1/sanity_checks_v1/SMALL_SAMPLE_OVERFIT_CURVE.png)",
        "",
        "## 3. Shuffled-label 随机对照",
        "",
        "Train 标签按 horizon 在各自有效训练样本内独立置换，保留输入、mask、训练流程和完整固定 Valid。随机界限为：|BAcc−0.5|≤0.02、|AUC−0.5|≤0.02、|MCC|≤0.04、|Rank IC|≤0.03。",
        "",
        "| seed | horizon | BAcc | AUC | MCC | Rank IC | 随机界限 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda x: (x["seed"], x["horizon"])):
        lines.append(f"| {row['seed']} | {row['horizon']} | {row['balanced_accuracy']:.4%} | {row['auc']:.4%} | {row['mcc']:.5f} | {row['rank_ic_mean']:.5f} | {'PASS' if row['within_random_bounds'] else 'FAIL'} |")
    lines += [
        "",
        "### Shuffled-label 结果解释",
        "",
        f"预注册的逐 seed 严格门结果为 **{posthoc['cells_within_random_bounds']}/{posthoc['total_cells']}** 个单元通过。越界单元为 seed 20260831/480m 与 seed 20260833/15m；没有任何 horizon 在两个或更多 seed 重复越界。因此只能记为 `FAIL_SHUFFLED_LABEL_RANDOM_CONTROL`，但当前没有形成跨 seed 可重复的泄漏模式。该一致性分析是结果后的解释，不修改原门。",
        "",
        "还观察到若干置换模型的 prediction up-rate 接近 0，说明 500-step 随机标签训练容易输出近常数；因此本检查主要依据 BAcc/AUC/MCC/Rank IC，而不能把普通 ACC 当随机性证据。",
        "",
        "## 解释边界",
        "",
        "- 三项全 PASS 只排除明显的 Label 计算、模型可训练性与标签/身份泄漏问题，不证明正式模型具有泛化能力。",
        "- 任一项 FAIL 都应先修复实现或数据链路，停止继续调整网络与损失。",
        "- 本报告不改变原正式模型的基线门失败结论。",
        "",
    ]
    doc = ROOT / contract["outputs"]["documentation"]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("label", "overfit", "shuffle", "aggregate"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "label":
        result = run_label_recomputation()
    elif args.mode == "overfit":
        result = run_small_sample_overfit()
    elif args.mode == "shuffle":
        if args.seed is None:
            raise SanityCheckError("--seed is required for shuffled-label")
        result = run_shuffled_label(args.seed)
    else:
        result = aggregate()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
