"""Continuous end-to-end Courage Strict direct-return training experiment.

This independent experiment reuses the accepted V1 provider and base
PatchTST architecture without mutating either.  It trains one continuous
optimizer/scheduler, applies deterministic horizon-specific target density,
and validates the complete rolling-origin population every 250 steps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset

from qlib.contrib.data.courage_strict_v1.dataset import CourageStrictV1Dataset
from qlib.contrib.model.courage_strict_v1 import _forward, _move
from qlib.contrib.model.patchtst_courage_strict_v1 import (
    HORIZONS_V1,
    PatchTSTCourageStrictV1,
)


class CourageStrictContinuousError(RuntimeError):
    """Raised when the bounded continuous experiment contract drifts."""


@dataclass(frozen=True)
class ContinuousConfig:
    project_root: Path
    provider_root: Path
    output_root: Path
    p0_manifest: Path
    origins: dict[str, dict[str, Any]]
    source: dict[str, Any]
    config_path: Path


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


def atomic_torch(path: Path, value: Any) -> str:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def load_config(path: str | Path) -> ContinuousConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "courage_strict_continuous_v1_config_v1":
        raise CourageStrictContinuousError("config schema drift")
    if raw.get("purpose") != "BOUNDED_CONTINUOUS_END_TO_END_DIRECT_RETURN_TRAINING":
        raise CourageStrictContinuousError("experiment purpose drift")
    roots = raw.get("roots", {})
    project_root = Path(roots.get("project_root", "")).resolve()
    if project_root != config_path.parents[2]:
        raise CourageStrictContinuousError("project root drift")
    training = raw.get("training", {})
    eval50_experiments = {
        "courage_strict_continuous_pit_10_15_v1",
        "courage_strict_continuous_pit_10_15_source_semantics_v2",
        "courage_strict_continuous_pit_10_15_source_semantics_v2_mse_v1",
    }
    validation_interval = 50 if raw.get("experiment_id") in eval50_experiments else 250
    expected = {
        "world_size": 8,
        "batch_size_per_rank": 128,
        "global_batch_size": 1024,
        "workers_per_rank": 8,
        "precision": "BF16",
        "epochs": 1,
        "peak_learning_rate": 0.00015,
        "warmup_fraction": 0.05,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "huber_delta": 1.0,
        "seed": 20260821,
        "validation_interval_steps": validation_interval,
        "materialized_signal_stride": 5,
        "effective_strides": {
            "short_5_15_30_60": 5,
            "mid_120": 15,
            "long_240_480": 30,
        },
    }
    loss_function = str(training.get("loss_function", "HUBER")).upper()
    if loss_function == "MSE":
        expected["loss_function"] = "MSE"
    elif loss_function != "HUBER":
        raise CourageStrictContinuousError("unsupported loss function")
    if training != expected:
        raise CourageStrictContinuousError("training parameter drift")
    diagnostic_experiment = (
        raw.get("experiment_id") == "courage_strict_continuous_v1_diagnostic_v1"
    )
    expected_diagnostics = {
        "enabled": True,
        "scale": "RAW_RETURN_AFTER_INVERSE_TRAIN_SCALER",
        "train_window_moments": True,
        "full_fixed_valid_moments": True,
        "statistics": [
            "prediction_mean",
            "prediction_std",
            "target_mean",
            "target_std",
            "bias",
            "prediction_up_rate",
            "target_up_rate",
            "prediction_zero_rate",
            "target_zero_rate",
        ],
    }
    if diagnostic_experiment:
        if raw.get("diagnostics") != expected_diagnostics:
            raise CourageStrictContinuousError("diagnostic logging contract drift")
    elif "diagnostics" in raw:
        raise CourageStrictContinuousError("diagnostics require a versioned experiment")
    model = raw.get("model", {})
    if not (
        model.get("end_to_end_joint_training") is True
        and model.get("freeze_backbone") is False
        and model.get("grouped_adapters") is False
        and model.get("market_residual_decomposition") is False
        and model.get("probability_head") is False
    ):
        raise CourageStrictContinuousError("model boundary drift")
    expected_selection_metric = f"valid_equal_head_standardized_{loss_function.lower()}"
    if raw.get("selection", {}).get("metric") != expected_selection_metric:
        raise CourageStrictContinuousError("selection metric/loss drift")
    authority = raw.get("authority", {})
    required_true = {
        "train_read_authorized",
        "rolling_valid_read_authorized",
        "training_authorized",
        "checkpoint_write_authorized",
    }
    if any(authority.get(name) is not True for name in required_true):
        raise CourageStrictContinuousError("required authority absent")
    forbidden = set(authority) - required_true
    if any(authority.get(name) is not False for name in forbidden):
        raise CourageStrictContinuousError("forbidden authority drift")
    origins = raw.get("origins", {})
    if tuple(origins) != (
        "origin_2026_01",
        "origin_2026_02",
        "origin_2026_03",
    ):
        raise CourageStrictContinuousError("rolling-origin identity drift")
    return ContinuousConfig(
        project_root=project_root,
        provider_root=(project_root / roots["provider_root"]).resolve(),
        output_root=(project_root / roots["output_root"]).resolve(),
        p0_manifest=(project_root / roots["p0_manifest"]).resolve(),
        origins=dict(origins),
        source=raw,
        config_path=config_path,
    )


def validate_identity(config: ContinuousConfig) -> dict[str, str]:
    expected = config.source["identity"]
    paths = {
        "provider_catalog_sha256": config.provider_root
        / "_courage_strict_v1_qlib_catalog.json",
        "p0_manifest_sha256": config.p0_manifest,
        "patchtst_module_sha256": config.project_root
        / "qlib/contrib/model/patchtst_courage_strict_v1.py",
    }
    candidate = config.source.get("membership_candidate")
    if candidate is not None:
        paths.update(
            {
                "candidate_membership_sha256": config.project_root
                / candidate["membership"],
                "candidate_membership_manifest_sha256": config.project_root
                / candidate["membership_manifest"],
                "candidate_contract_sha256": config.project_root
                / candidate["contract"],
            }
        )
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != expected:
        raise CourageStrictContinuousError(
            f"input identity drift: observed={observed} expected={expected}"
        )
    p0 = json.loads(config.p0_manifest.read_text(encoding="utf-8"))
    if (
        p0.get("decision") != "COMPLETE_CORRECTED_DATA_REPAIR_AND_FROZEN_REPLAY"
        or p0.get("march_parity", {}).get("decision")
        != "PASS_MARCH_CANONICAL_BUILDER_PREDICTION_METRIC_PARITY"
        or p0.get("rolling_origin_purge", {}).get("decision")
        != "PASS_ROLLING_ORIGIN_PER_HEAD_LABEL_MATURITY_PURGE"
        or p0.get("boundaries", {}).get("june_or_later_read") is not False
    ):
        raise CourageStrictContinuousError("P0 terminal gate failed")
    return observed


class StratifiedTimeCellDistributedSampler(Sampler[int]):
    """Interleave signal timestamps while shuffling stocks inside each cell."""

    def __init__(
        self,
        ends: np.ndarray,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        self.ends = np.asarray(ends, dtype=np.int64)
        if self.ends.ndim != 1 or not len(self.ends):
            raise CourageStrictContinuousError("sampler references are empty")
        if not 0 <= rank < num_replicas:
            raise CourageStrictContinuousError("invalid distributed sampler rank")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.start_offset = 0
        self.num_samples = math.ceil(len(self.ends) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        if not 0 <= offset <= self.num_samples:
            raise CourageStrictContinuousError("sampler resume offset drift")
        self.start_offset = int(offset)

    def _global_order(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + self.epoch)
        stable = np.argsort(self.ends, kind="stable")
        sorted_ends = self.ends[stable]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ends)) + 1, len(stable)]
        counts = np.diff(boundaries)
        rows = len(counts)
        width = int(counts.max())
        matrix = np.full((rows, width), -1, dtype=np.int32)
        for row, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            cell = stable[start:stop].copy()
            rng.shuffle(cell)
            matrix[row, : len(cell)] = cell.astype(np.int32, copy=False)
        matrix = matrix[rng.permutation(rows)]
        result = matrix.T.reshape(-1)
        result = result[result >= 0].astype(np.int64, copy=False)
        if len(result) != len(self.ends) or len(np.unique(result)) != len(result):
            raise CourageStrictContinuousError("stratified sampler permutation drift")
        if self.total_size > len(result):
            result = np.concatenate((result, result[: self.total_size - len(result)]))
        return result

    def __iter__(self) -> Iterator[int]:
        indices = self._global_order()[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise CourageStrictContinuousError("rank sampler cardinality drift")
        return iter(indices[self.start_offset :].tolist())

    def __len__(self) -> int:
        return self.num_samples - self.start_offset


def effective_target_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply fixed 5/15/30-minute head density on the stride-5 provider."""

    base = batch["target_mask"].clone()
    slot = batch["minute_index"][:, -1]
    session = batch["session_index"][:, -1]
    phase = torch.where(session.eq(0), slot, slot - 121)
    mid = phase.remainder(15).eq(0)
    long = phase.remainder(30).eq(0)
    base[:, 4] &= mid
    base[:, 5:] &= long[:, None]
    return base


def optimizer_parameter_groups(
    model: torch.nn.Module, *, weight_decay: float
) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if (
            parameter.ndim < 2
            or lowered.endswith(".bias")
            or "norm" in lowered
            or "embedding" in lowered
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay or not no_decay:
        raise CourageStrictContinuousError("optimizer parameter grouping is degenerate")
    if len({id(item) for item in decay + no_decay}) != len(decay) + len(no_decay):
        raise CourageStrictContinuousError("optimizer parameter duplication")
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def continuous_scheduler(
    optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_fraction: float
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 1 or not 0 < warmup_fraction < 1:
        raise CourageStrictContinuousError("invalid scheduler geometry")
    warmup = max(1, math.ceil(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-8)
        progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 8 or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise CourageStrictContinuousError("formal execution requires exactly 8 GPUs")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    return rank, world, local, torch.device("cuda", local)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(
    data: Any,
    *,
    train: bool,
    rank: int,
    world: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, StratifiedTimeCellDistributedSampler | None]:
    sampler = None
    selected = data
    if train:
        raw = data.raw if hasattr(data, "raw") else data
        sampler = StratifiedTimeCellDistributedSampler(
            raw.ends, num_replicas=world, rank=rank, seed=seed
        )
    else:
        selected = Subset(data, range(rank, len(data), world))
    loader = DataLoader(
        selected,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=False,
    )
    return loader, sampler


def _masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    world: int,
    loss_function: str = "HUBER",
) -> torch.Tensor:
    errors = _pointwise_loss(prediction, target, loss_function=loss_function)
    numerator = (errors * mask).sum(dim=0)
    denominator = mask.sum(dim=0, dtype=torch.float64)
    dist.all_reduce(denominator)
    if (denominator <= 0).any():
        raise CourageStrictContinuousError("optimizer batch contains empty head")
    return (world * numerator / denominator.to(numerator.dtype)).mean()


def _pointwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_function: str,
) -> torch.Tensor:
    name = str(loss_function).upper()
    if name == "HUBER":
        return functional.huber_loss(
            prediction.float(), target.float(), delta=1.0, reduction="none"
        )
    if name == "MSE":
        return functional.mse_loss(prediction.float(), target.float(), reduction="none")
    raise CourageStrictContinuousError(f"unsupported loss function: {loss_function}")


def _reduce_metrics(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> tuple[list[float], float, list[int]]:
    left = numerator.clone()
    right = denominator.clone()
    dist.all_reduce(left)
    dist.all_reduce(right)
    if (right <= 0).any():
        raise CourageStrictContinuousError("metric contains empty head")
    values = (left / right).cpu().tolist()
    return (
        [float(value) for value in values],
        float(np.mean(values)),
        [int(value) for value in right.cpu().tolist()],
    )


_MOMENT_COLUMNS = (
    "rows",
    "prediction_sum",
    "prediction_square_sum",
    "target_sum",
    "target_square_sum",
    "error_sum",
    "prediction_up_rows",
    "target_up_rows",
    "prediction_zero_rows",
    "target_zero_rows",
)


def _target_scaler_tensors(
    scalers: dict[str, Any], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    medians = torch.tensor(
        [scalers["targets"][str(horizon)]["median"] for horizon in HORIZONS_V1],
        dtype=torch.float64,
        device=device,
    )
    iqrs = torch.tensor(
        [scalers["targets"][str(horizon)]["iqr"] for horizon in HORIZONS_V1],
        dtype=torch.float64,
        device=device,
    )
    if (
        not torch.isfinite(medians).all()
        or not torch.isfinite(iqrs).all()
        or (iqrs <= 0).any()
    ):
        raise CourageStrictContinuousError("invalid target scaler for diagnostics")
    return medians, iqrs


def _raw_moment_totals(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_medians: torch.Tensor,
    target_iqrs: torch.Tensor,
) -> torch.Tensor:
    active = mask.bool()
    raw_prediction = prediction.double() * target_iqrs + target_medians
    raw_target = target.double() * target_iqrs + target_medians
    weights = active.double()
    return torch.stack(
        (
            weights.sum(dim=0),
            (raw_prediction * weights).sum(dim=0),
            (raw_prediction.square() * weights).sum(dim=0),
            (raw_target * weights).sum(dim=0),
            (raw_target.square() * weights).sum(dim=0),
            ((raw_prediction - raw_target) * weights).sum(dim=0),
            ((raw_prediction > 0) & active).sum(dim=0).double(),
            ((raw_target > 0) & active).sum(dim=0).double(),
            ((raw_prediction == 0) & active).sum(dim=0).double(),
            ((raw_target == 0) & active).sum(dim=0).double(),
        ),
        dim=1,
    )


def _moment_summary(totals: torch.Tensor) -> dict[str, dict[str, float | int]]:
    values = totals.detach().cpu().numpy().astype(np.float64, copy=False)
    if values.shape != (len(HORIZONS_V1), len(_MOMENT_COLUMNS)):
        raise CourageStrictContinuousError("diagnostic moment shape drift")
    result: dict[str, dict[str, float | int]] = {}
    for head, horizon in enumerate(HORIZONS_V1):
        row = dict(zip(_MOMENT_COLUMNS, values[head], strict=True))
        count = int(row["rows"])
        if count <= 0:
            raise CourageStrictContinuousError("diagnostic moment contains empty head")
        prediction_mean = row["prediction_sum"] / count
        target_mean = row["target_sum"] / count
        prediction_variance = max(
            row["prediction_square_sum"] / count - prediction_mean**2, 0.0
        )
        target_variance = max(row["target_square_sum"] / count - target_mean**2, 0.0)
        result[str(horizon)] = {
            "rows": count,
            "prediction_mean": float(prediction_mean),
            "prediction_std": float(math.sqrt(prediction_variance)),
            "target_mean": float(target_mean),
            "target_std": float(math.sqrt(target_variance)),
            "bias": float(row["error_sum"] / count),
            "prediction_up_rate": float(row["prediction_up_rows"] / count),
            "target_up_rate": float(row["target_up_rows"] / count),
            "prediction_zero_rate": float(row["prediction_zero_rows"] / count),
            "target_zero_rate": float(row["target_zero_rows"] / count),
        }
    return result


def _reduce_moment_summary(totals: torch.Tensor) -> dict[str, dict[str, float | int]]:
    reduced = totals.clone()
    dist.all_reduce(reduced)
    return _moment_summary(reduced)


@torch.no_grad()
def evaluate_valid(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_function: str = "HUBER",
    target_scalers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> (
    tuple[list[float], float, list[int]]
    | tuple[list[float], float, list[int], dict[str, dict[str, float | int]]]
):
    was_training = model.training
    model.eval()
    numerator = torch.zeros(len(HORIZONS_V1), dtype=torch.float64, device=device)
    denominator = torch.zeros_like(numerator)
    moments = (
        torch.zeros(
            (len(HORIZONS_V1), len(_MOMENT_COLUMNS)),
            dtype=torch.float64,
            device=device,
        )
        if target_scalers is not None
        else None
    )
    for batch in loader:
        moved = _move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = _forward(model, moved)
        errors = _pointwise_loss(
            prediction,
            moved["targets"],
            loss_function=loss_function,
        )
        numerator += (errors * moved["target_mask"]).sum(dim=0).double()
        denominator += moved["target_mask"].sum(dim=0).double()
        if moments is not None and target_scalers is not None:
            moments += _raw_moment_totals(
                prediction,
                moved["targets"],
                moved["target_mask"],
                target_medians=target_scalers[0],
                target_iqrs=target_scalers[1],
            )
    result = _reduce_metrics(numerator, denominator)
    diagnostics = _reduce_moment_summary(moments) if moments is not None else None
    if was_training:
        model.train()
    if diagnostics is not None:
        return (*result, diagnostics)
    return result


def _window_coverage(
    records: dict[str, set[int]], *, rank: int, world: int
) -> dict[str, Any] | None:
    local = {name: sorted(values) for name, values in records.items()}
    gathered: list[dict[str, list[int]] | None] = [None] * world
    dist.all_gather_object(gathered, local)
    if rank != 0:
        return None
    merged = {
        name: set().union(*(set(item[name]) for item in gathered if item is not None))
        for name in local
    }
    return {
        "unique_dates": len(merged["dates"]),
        "unique_instruments": len(merged["instruments"]),
        "unique_signal_slots": len(merged["slots"]),
        "sessions": sorted(merged["sessions"]),
        "unique_row_keys": len(merged["rows"]),
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scalers: dict[str, Any],
    origin: str,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    samples_seen: int,
    valid_metric: float,
    config_sha256: str,
    identity: dict[str, str],
    experiment_id: str,
    loss_function: str,
) -> dict[str, Any]:
    base = model.module if isinstance(model, DistributedDataParallel) else model
    metric_key = f"valid_equal_head_standardized_{loss_function.lower()}"
    return {
        "schema_version": "courage_strict_continuous_checkpoint_v1",
        "experiment_id": experiment_id,
        "origin": origin,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "loss_function": loss_function,
        metric_key: valid_metric,
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in base.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scalers": scalers,
        "industry_vocab_size": base.industry_vocab_size,
        "config_sha256": config_sha256,
        "input_identity": identity,
        "backbone_frozen": False,
        "optimizer_restarted": False,
        "june_or_later_read": False,
    }


def _load_scalers(
    config: ContinuousConfig, *, origin: str, train_data: Any
) -> tuple[dict[str, Any], str]:
    item = config.origins[origin]
    path = (config.project_root / item["accepted_scaler"]).resolve()
    observed = sha256_file(path)
    if observed != item["accepted_scaler_file_sha256"]:
        raise CourageStrictContinuousError(f"accepted scaler byte drift: {origin}")
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = train_data.raw if hasattr(train_data, "raw") else train_data
    reference = hashlib.sha256(
        raw.symbol_positions.tobytes(order="C") + raw.ends.tobytes(order="C")
    ).hexdigest()
    population_policy = config.source.get("scaler_policy", "exact_population")
    exact_population = population_policy == "exact_population"
    fixed_baseline = (
        population_policy == "reuse_baseline_train_scaler_fixed_preprocessing"
    )
    if not (exact_population or fixed_baseline):
        raise CourageStrictContinuousError("unknown scaler population policy")
    if (
        value.get("fit_role") != "train_only"
        or value.get("train_segment") != item["train"]
        or value.get("provider_catalog_sha256")
        != config.source["identity"]["provider_catalog_sha256"]
    ):
        raise CourageStrictContinuousError(
            f"accepted scaler population drift: {origin}"
        )
    if exact_population and (
        int(value.get("train_samples", -1)) != len(raw)
        or value.get("train_reference_sha256") != reference
    ):
        raise CourageStrictContinuousError(
            f"accepted scaler population drift: {origin}"
        )
    return value, observed


def _gradient_summary(
    values: Sequence[float], *, rank: int, world: int, clip_norm: float
) -> dict[str, float] | None:
    gathered: list[list[float] | None] = [None] * world
    dist.all_gather_object(gathered, list(values))
    if rank != 0:
        return None
    merged = np.asarray(
        [value for part in gathered if part is not None for value in part],
        dtype=np.float64,
    )
    if not len(merged) or not np.isfinite(merged).all():
        raise CourageStrictContinuousError("gradient window is invalid")
    return {
        "median_pre_clip": float(np.median(merged)),
        "p95_pre_clip": float(np.quantile(merged, 0.95)),
        "max_pre_clip": float(merged.max()),
        "clip_trigger_ratio": float((merged > clip_norm).mean()),
    }


def _reset_coverage() -> dict[str, set[int]]:
    return {
        "dates": set(),
        "instruments": set(),
        "slots": set(),
        "sessions": set(),
        "rows": set(),
    }


def _update_coverage(
    coverage: dict[str, set[int]], batch: dict[str, torch.Tensor], *, calendar_size: int
) -> None:
    calendar = batch["calendar_index"].numpy().astype(np.int64, copy=False)
    symbols = batch["symbol_position"].numpy().astype(np.int64, copy=False)
    slots = batch["minute_index"][:, -1].numpy().astype(np.int64, copy=False)
    sessions = batch["session_index"][:, -1].numpy().astype(np.int64, copy=False)
    coverage["dates"].update((calendar // 240).tolist())
    coverage["instruments"].update(symbols.tolist())
    coverage["slots"].update(slots.tolist())
    coverage["sessions"].update(sessions.tolist())
    coverage["rows"].update((symbols * int(calendar_size) + calendar).tolist())


def train_origin(
    config_path: str | Path,
    *,
    origin: str,
    output_override: Path | None = None,
    maximum_steps: int | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any] | None:
    """Train one rolling origin under the exact continuous contract.

    ``maximum_steps`` is reserved for bounded smoke runs. Formal training must
    omit it and completes exactly one sampler pass.
    """

    config = load_config(config_path)
    identity = validate_identity(config)
    if origin not in config.origins:
        raise CourageStrictContinuousError(f"unknown origin: {origin}")
    rank, world, local, device = _distributed()
    training = config.source["training"]
    loss_function = str(training.get("loss_function", "HUBER")).upper()
    metric_suffix = loss_function.lower()
    valid_metric_key = f"valid_equal_head_standardized_{metric_suffix}"
    valid_per_head_key = f"valid_per_head_standardized_{metric_suffix}"
    train_metric_key = f"train_window_equal_head_standardized_{metric_suffix}"
    train_per_head_key = f"train_window_per_head_standardized_{metric_suffix}"
    _seed(int(training["seed"]) + rank)
    origin_spec = config.origins[origin]
    output = (
        output_override.resolve()
        if output_override is not None
        else (config.output_root / origin).resolve()
    )
    if rank == 0:
        if resume_from is None:
            if output.exists():
                raise CourageStrictContinuousError(f"output already exists: {output}")
            (output / "checkpoints").mkdir(parents=True)
        elif not output.is_dir() or resume_from.resolve().parent != output:
            raise CourageStrictContinuousError("resume output/path identity drift")
    dist.barrier()

    dataset = CourageStrictV1Dataset(
        provider_root=config.provider_root,
        segments={
            "train": tuple(origin_spec["train"]),
            "valid": tuple(origin_spec["valid"]),
        },
        turnover_band=(
            tuple(config.source["membership_candidate"]["turnover_band"])
            if "membership_candidate" in config.source
            else None
        ),
    )
    raw_train = dataset.prepare("train", cache_capacity=8)
    # Validate the accepted deterministic Train-only statistics before binding
    # them to the scaled dataset. No learned model/optimizer state is reused.
    placeholder = type("RawHolder", (), {"raw": raw_train})()
    scalers, scaler_file_sha = _load_scalers(
        config, origin=origin, train_data=placeholder
    )
    train_data = dataset.prepare("train", scalers=scalers, cache_capacity=8)
    valid_data = dataset.prepare("valid", scalers=scalers, cache_capacity=8)
    train_loader, sampler = _loader(
        train_data,
        train=True,
        rank=rank,
        world=world,
        batch_size=int(training["batch_size_per_rank"]),
        workers=int(training["workers_per_rank"]),
        seed=int(training["seed"]),
    )
    valid_loader, valid_sampler = _loader(
        valid_data,
        train=False,
        rank=rank,
        world=world,
        batch_size=int(training["batch_size_per_rank"]),
        workers=int(training["workers_per_rank"]),
        seed=int(training["seed"]),
    )
    if sampler is None or valid_sampler is not None:
        raise CourageStrictContinuousError("loader sampler contract drift")
    steps_per_epoch = math.ceil(
        sampler.num_samples / int(training["batch_size_per_rank"])
    )
    total_steps = steps_per_epoch * int(training["epochs"])
    if maximum_steps is not None:
        if not 1 <= maximum_steps <= total_steps:
            raise CourageStrictContinuousError("invalid bounded maximum_steps")
        total_steps = int(maximum_steps)
    vocabulary = json.loads(
        (config.provider_root / "industry_vocabulary.json").read_text(encoding="utf-8")
    )
    network = PatchTSTCourageStrictV1(industry_vocab_size=len(vocabulary)).to(device)
    model: torch.nn.Module = DistributedDataParallel(
        network, device_ids=[local], output_device=local
    )
    groups = optimizer_parameter_groups(
        model, weight_decay=float(training["weight_decay"])
    )
    optimizer = torch.optim.AdamW(groups, lr=float(training["peak_learning_rate"]))
    scheduler = continuous_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=float(training["warmup_fraction"]),
    )
    config_sha = sha256_file(config.config_path)
    diagnostics_enabled = bool(
        config.source.get("diagnostics", {}).get("enabled", False)
    )
    diagnostic_target_scalers = (
        _target_scaler_tensors(scalers, device=device) if diagnostics_enabled else None
    )
    interval = int(training["validation_interval_steps"])
    validation_steps = set(range(interval, total_steps + 1, interval)) | {total_steps}
    history: list[dict[str, Any]] = []
    best_metric = math.inf
    best_step = -1
    global_step = 0
    samples_seen = 0
    start_batch_in_epoch = 0
    if resume_from is not None:
        checkpoint = torch.load(
            resume_from.resolve(), map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("schema_version")
            != "courage_strict_continuous_checkpoint_v1"
            or checkpoint.get("origin") != origin
            or checkpoint.get("config_sha256") != config_sha
            or checkpoint.get("input_identity") != identity
            or checkpoint.get("backbone_frozen") is not False
            or checkpoint.get("optimizer_restarted") is not False
        ):
            raise CourageStrictContinuousError("resume checkpoint identity drift")
        network.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        global_step = int(checkpoint["global_step"])
        samples_seen = int(checkpoint["samples_seen"])
        start_batch_in_epoch = int(checkpoint["batch_in_epoch"])
        if not 0 < global_step < total_steps or start_batch_in_epoch != global_step:
            raise CourageStrictContinuousError("resume progress drift")
        sampler.set_start_offset(
            start_batch_in_epoch * int(training["batch_size_per_rank"])
        )
        history_path = output / "loss_curve.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best_value = json.loads((output / "best.json").read_text(encoding="utf-8"))
        best_metric = float(best_value[valid_metric_key])
        best_step = int(best_value["global_step"])
    started = time.time()
    window_started = time.perf_counter()
    window_num = torch.zeros(len(HORIZONS_V1), dtype=torch.float64, device=device)
    window_den = torch.zeros_like(window_num)
    window_moments = (
        torch.zeros(
            (len(HORIZONS_V1), len(_MOMENT_COLUMNS)),
            dtype=torch.float64,
            device=device,
        )
        if diagnostics_enabled
        else None
    )
    window_gradients: list[float] = []
    window_samples_local = 0
    coverage = _reset_coverage()
    cumulative_dates: set[int] = set()
    cumulative_instruments: set[int] = set()
    cumulative_slots: set[int] = set()
    cumulative_sessions: set[int] = set()
    sampler.set_epoch(1)
    model.train()
    for batch_in_epoch, batch in enumerate(
        train_loader, start=start_batch_in_epoch + 1
    ):
        if global_step >= total_steps:
            break
        _update_coverage(
            coverage, batch, calendar_size=len(raw_train.provider.calendar)
        )
        moved = _move(batch, device)
        mask = effective_target_mask(moved)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = _forward(model, moved)
            loss = _masked_loss(
                prediction,
                moved["targets"],
                mask,
                world=world,
                loss_function=loss_function,
            )
        if not torch.isfinite(loss):
            raise CourageStrictContinuousError("non-finite training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        global_step += 1
        window_gradients.append(float(grad_norm.detach().cpu()))
        window_samples_local += int(moved["targets"].shape[0])
        with torch.no_grad():
            errors = _pointwise_loss(
                prediction,
                moved["targets"],
                loss_function=loss_function,
            )
            window_num += (errors * mask).sum(dim=0).double()
            window_den += mask.sum(dim=0).double()
            if window_moments is not None and diagnostic_target_scalers is not None:
                window_moments += _raw_moment_totals(
                    prediction,
                    moved["targets"],
                    mask,
                    target_medians=diagnostic_target_scalers[0],
                    target_iqrs=diagnostic_target_scalers[1],
                )

        if global_step not in validation_steps:
            continue
        elapsed = torch.tensor(
            [time.perf_counter() - window_started], dtype=torch.float64, device=device
        )
        local_samples = torch.tensor(
            [window_samples_local], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(local_samples, op=dist.ReduceOp.SUM)
        window_per_head, window_metric, effective_counts = _reduce_metrics(
            window_num, window_den
        )
        train_prediction_diagnostics = (
            _reduce_moment_summary(window_moments)
            if window_moments is not None
            else None
        )
        gradient = _gradient_summary(
            window_gradients,
            rank=rank,
            world=world,
            clip_norm=float(training["gradient_clip_norm"]),
        )
        coverage_result = _window_coverage(coverage, rank=rank, world=world)
        cumulative_dates.update(coverage["dates"])
        cumulative_instruments.update(coverage["instruments"])
        cumulative_slots.update(coverage["slots"])
        cumulative_sessions.update(coverage["sessions"])
        samples_seen += int(local_samples.item())
        dist.barrier()
        valid_result = evaluate_valid(
            model,
            valid_loader,
            device=device,
            loss_function=loss_function,
            target_scalers=diagnostic_target_scalers,
        )
        if diagnostics_enabled:
            (
                valid_per_head,
                valid_metric,
                valid_counts,
                valid_prediction_diagnostics,
            ) = valid_result
        else:
            valid_per_head, valid_metric, valid_counts = valid_result
            valid_prediction_diagnostics = None
        dist.barrier()
        if rank == 0:
            if gradient is None or coverage_result is None:
                raise CourageStrictContinuousError("rank-zero diagnostics absent")
            record = {
                "global_step": global_step,
                "epoch": 1,
                "batch_in_epoch": batch_in_epoch,
                "samples_seen": samples_seen,
                "sample_fraction_of_train": samples_seen / len(raw_train),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "loss_function": loss_function,
                train_metric_key: window_metric,
                train_per_head_key: dict(
                    zip(map(str, HORIZONS_V1), window_per_head, strict=True)
                ),
                "train_window_effective_label_counts": dict(
                    zip(map(str, HORIZONS_V1), effective_counts, strict=True)
                ),
                valid_metric_key: valid_metric,
                valid_per_head_key: dict(
                    zip(map(str, HORIZONS_V1), valid_per_head, strict=True)
                ),
                "valid_label_counts": dict(
                    zip(map(str, HORIZONS_V1), valid_counts, strict=True)
                ),
                "gradient": gradient,
                "window_coverage": coverage_result,
                "cumulative_coverage": {
                    "unique_dates_local_union": len(cumulative_dates),
                    "unique_instruments_local_union": len(cumulative_instruments),
                    "unique_signal_slots_local_union": len(cumulative_slots),
                    "sessions_local_union": sorted(cumulative_sessions),
                },
                "window_samples_per_second": float(
                    local_samples.item() / elapsed.item()
                ),
            }
            if diagnostics_enabled:
                if (
                    train_prediction_diagnostics is None
                    or valid_prediction_diagnostics is None
                ):
                    raise CourageStrictContinuousError(
                        "rank-zero prediction diagnostics absent"
                    )
                record["diagnostic_scale"] = "RAW_RETURN_AFTER_INVERSE_TRAIN_SCALER"
                record["train_window_prediction_diagnostics"] = (
                    train_prediction_diagnostics
                )
                record["valid_prediction_diagnostics"] = valid_prediction_diagnostics
            history.append(record)
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scalers=scalers,
                origin=origin,
                epoch=1,
                batch_in_epoch=batch_in_epoch,
                global_step=global_step,
                samples_seen=samples_seen,
                valid_metric=valid_metric,
                config_sha256=config_sha,
                identity=identity,
                experiment_id=config.source["experiment_id"],
                loss_function=loss_function,
            )
            atomic_torch(output / "last.pt", payload)
            if valid_metric < best_metric:
                best_metric = valid_metric
                best_step = global_step
                best_sha = atomic_torch(output / "best.pt", payload)
                atomic_json(
                    output / "best.json",
                    {
                        "global_step": global_step,
                        "samples_seen": samples_seen,
                        "loss_function": loss_function,
                        valid_metric_key: valid_metric,
                        valid_per_head_key: record[valid_per_head_key],
                        "checkpoint_sha256": best_sha,
                        "selection_rule": "strict_minimum_earlier_tie",
                    },
                )
            atomic_json(output / "loss_curve.json", history)
        window_num.zero_()
        window_den.zero_()
        if window_moments is not None:
            window_moments.zero_()
        window_gradients.clear()
        window_samples_local = 0
        coverage = _reset_coverage()
        window_started = time.perf_counter()
        model.train()

    dist.barrier()
    if rank == 0:
        manifest = {
            "schema_version": "courage_strict_continuous_training_manifest_v1",
            "decision": "PASS_CONTINUOUS_ORIGIN_TRAINING_COMPLETE",
            "experiment_id": config.source["experiment_id"],
            "membership_candidate": config.source.get("membership_candidate"),
            "scaler_policy": config.source.get("scaler_policy", "exact_population"),
            "origin": origin,
            "train_segment": origin_spec["train"],
            "valid_segment": origin_spec["valid"],
            "train_rows": len(raw_train),
            "valid_rows": len(valid_data),
            "steps_per_epoch": steps_per_epoch,
            "steps_executed": global_step,
            "samples_seen": samples_seen,
            "loss_function": loss_function,
            "best_step": best_step,
            f"best_{valid_metric_key}": best_metric,
            "best_checkpoint_sha256": sha256_file(output / "best.pt"),
            "last_checkpoint_sha256": sha256_file(output / "last.pt"),
            "accepted_scaler_file_sha256": scaler_file_sha,
            "config_sha256": config_sha,
            "input_identity": identity,
            "runtime_seconds": time.time() - started,
            "maximum_steps_override": maximum_steps,
            "resumed_from": (
                str(resume_from.resolve().relative_to(config.project_root))
                if resume_from is not None
                else None
            ),
            "backbone_frozen": False,
            "optimizer_restarted": False,
            "june_or_later_read": False,
            "refit_executed": False,
            "strategy_or_backtest_executed": False,
            "trading_executed": False,
            "remote_push_executed": False,
        }
        atomic_json(output / "_training_manifest.json", manifest)
    else:
        manifest = None
    dist.barrier()
    dist.destroy_process_group()
    return manifest


__all__ = [
    "ContinuousConfig",
    "CourageStrictContinuousError",
    "StratifiedTimeCellDistributedSampler",
    "continuous_scheduler",
    "effective_target_mask",
    "load_config",
    "optimizer_parameter_groups",
    "train_origin",
    "validate_identity",
]
