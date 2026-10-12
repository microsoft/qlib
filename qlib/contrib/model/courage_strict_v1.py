"""Qlib Model wrapper for the Courage Strict V1 multi-horizon PatchTST."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from qlib.contrib.data.courage_strict_v1.dataset import (
    CourageStrictV1Dataset,
    fit_train_scalers,
)
from qlib.contrib.model.patchtst_courage_strict_v1 import (
    HORIZONS_V1,
    PatchTSTCourageStrictV1,
)
from qlib.model.base import Model


class CourageStrictV1TrainingError(RuntimeError):
    """Raised when a frozen V1 training invariant is violated."""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True) for key, value in batch.items()
    }


def _forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["values"],
        batch["available"],
        batch["missing"],
        batch["padding"],
        batch["minute_index"],
        batch["session_index"],
        batch["slow_values"],
        batch["slow_available"],
        batch["industry_id"],
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, *, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total_steps * 0.05))

    def factor(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-8)
        progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _equal_head_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    world_size: int,
) -> torch.Tensor:
    errors = functional.huber_loss(
        prediction.float(), target.float(), delta=1.0, reduction="none"
    )
    numerator = (errors * mask).sum(dim=0)
    denominator = mask.sum(dim=0, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    active = denominator.gt(0)
    if not active.any():
        raise CourageStrictV1TrainingError("optimizer step has no active target")
    # DDP averages gradients, hence the world-size factor on local numerators.
    return (
        world_size * numerator[active] / denominator[active].to(dtype=numerator.dtype)
    ).mean()


class CourageStrictV1Model(Model):
    """Train and infer V1 directly from Qlib 1-minute bins.

    Launching ``fit`` under ``torchrun`` activates DDP. A normal Python process
    remains a valid single-device Qlib Model execution path.
    """

    def __init__(
        self,
        *,
        batch_size: int = 256,
        num_workers: int = 8,
        max_epochs: int = 6,
        min_epochs: int = 2,
        patience: int = 2,
        min_delta: float = 1e-4,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        seed: int = 20260818,
        output_dir: str = "artifacts/courage_strict_v1/training",
    ) -> None:
        if min_epochs < 1 or max_epochs < min_epochs or patience < 1:
            raise CourageStrictV1TrainingError("invalid epoch budget")
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_epochs = int(max_epochs)
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.output_dir = str(output_dir)
        self.fitted = False
        self.scalers: dict[str, Any] | None = None
        self.model_state: dict[str, torch.Tensor] | None = None
        self.industry_vocab_size: int | None = None
        self.history: list[dict[str, float | int]] = []

    @staticmethod
    def _distributed() -> tuple[int, int, int]:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
        return rank, world_size, local_rank

    @staticmethod
    def _device(local_rank: int) -> torch.device:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cpu")

    def _loader(
        self,
        data,
        *,
        train: bool,
        rank: int,
        world_size: int,
    ) -> DataLoader:
        sampler = None
        selected = data
        if world_size > 1:
            if train:
                sampler = DistributedSampler(
                    data,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=self.seed,
                    drop_last=False,
                )
            else:
                selected = Subset(data, range(rank, len(data), world_size))
        return DataLoader(
            selected,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=train and sampler is None,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=False,
        )

    @staticmethod
    def _industry_vocab(provider_root: Path) -> int:
        value = json.loads(
            (provider_root / "industry_vocabulary.json").read_text(encoding="utf-8")
        )
        if not isinstance(value, dict) or value.get("UNKNOWN") != 0:
            raise CourageStrictV1TrainingError("industry vocabulary drift")
        return len(value)

    def fit(
        self,
        dataset: CourageStrictV1Dataset,
        reweighter=None,
        evals_result: dict[str, Any] | None = None,
    ) -> None:
        del reweighter
        rank, world_size, local_rank = self._distributed()
        device = self._device(local_rank)
        _seed_everything(self.seed + rank)
        output = Path(self.output_dir).resolve()
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            train_raw = dataset.prepare("train", cache_capacity=8)
            scaler_path = output / "train_scalers.json"
            if scaler_path.is_file():
                scalers = json.loads(scaler_path.read_text(encoding="utf-8"))
                reference_sha = hashlib.sha256(
                    train_raw.symbol_positions.tobytes(order="C")
                    + train_raw.ends.tobytes(order="C")
                ).hexdigest()
                if (
                    scalers.get("train_segment") != [train_raw.start, train_raw.end]
                    or scalers.get("train_samples") != len(train_raw)
                    or scalers.get("train_reference_sha256") != reference_sha
                ):
                    raise CourageStrictV1TrainingError(
                        "existing scaler population drift"
                    )
            else:
                scalers = fit_train_scalers(train_raw)
                scaler_path.write_text(
                    json.dumps(scalers, ensure_ascii=False, sort_keys=True, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
        if world_size > 1:
            dist.barrier()
        self.scalers = json.loads(
            (output / "train_scalers.json").read_text(encoding="utf-8")
        )
        train_data = dataset.prepare("train", scalers=self.scalers, cache_capacity=8)
        valid_data = dataset.prepare("valid", scalers=self.scalers, cache_capacity=8)
        train_loader = self._loader(
            train_data, train=True, rank=rank, world_size=world_size
        )
        valid_loader = self._loader(
            valid_data, train=False, rank=rank, world_size=world_size
        )
        self.industry_vocab_size = self._industry_vocab(dataset.provider_root)
        network = PatchTSTCourageStrictV1(
            industry_vocab_size=self.industry_vocab_size
        ).to(device)
        model: torch.nn.Module = network
        if world_size > 1:
            model = DistributedDataParallel(
                network,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedule = _scheduler(
            optimizer, total_steps=max(1, self.max_epochs * len(train_loader))
        )
        best_metric = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        history: list[dict[str, float | int]] = []
        amp_enabled = device.type == "cuda"
        for epoch in range(1, self.max_epochs + 1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            model.train()
            train_numerator = torch.zeros(
                len(HORIZONS_V1), dtype=torch.float64, device=device
            )
            train_denominator = torch.zeros_like(train_numerator)
            for batch in train_loader:
                batch = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    prediction = _forward(model, batch)
                    loss = _equal_head_loss(
                        prediction,
                        batch["targets"],
                        batch["target_mask"],
                        world_size=world_size,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                with torch.no_grad():
                    errors = functional.huber_loss(
                        prediction.float(),
                        batch["targets"].float(),
                        delta=1.0,
                        reduction="none",
                    )
                    train_numerator += (
                        (errors * batch["target_mask"]).sum(dim=0).double()
                    )
                    train_denominator += batch["target_mask"].sum(dim=0).double()
            model.eval()
            valid_numerator = torch.zeros_like(train_numerator)
            valid_denominator = torch.zeros_like(train_denominator)
            with torch.no_grad():
                for batch in valid_loader:
                    batch = _move(batch, device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        prediction = _forward(model, batch)
                    errors = functional.huber_loss(
                        prediction.float(),
                        batch["targets"].float(),
                        delta=1.0,
                        reduction="none",
                    )
                    valid_numerator += (
                        (errors * batch["target_mask"]).sum(dim=0).double()
                    )
                    valid_denominator += batch["target_mask"].sum(dim=0).double()
            if world_size > 1:
                for value in (
                    train_numerator,
                    train_denominator,
                    valid_numerator,
                    valid_denominator,
                ):
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
            if (valid_denominator <= 0).any() or (train_denominator <= 0).any():
                raise CourageStrictV1TrainingError("epoch contains an empty head")
            train_metric = float((train_numerator / train_denominator).mean().item())
            valid_metric = float((valid_numerator / valid_denominator).mean().item())
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_equal_head_standardized_huber": train_metric,
                "valid_equal_head_standardized_huber": valid_metric,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(record)
            improved = valid_metric < best_metric - self.min_delta
            if improved:
                best_metric = valid_metric
                stale = 0
                unwrapped = (
                    model.module
                    if isinstance(model, DistributedDataParallel)
                    else model
                )
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in unwrapped.state_dict().items()
                }
            else:
                stale += 1
            if rank == 0:
                (output / "training_history.json").write_text(
                    json.dumps(history, ensure_ascii=False, sort_keys=True, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                if improved and best_state is not None:
                    torch.save(
                        {
                            "schema_version": "courage_strict_v1_checkpoint_v1",
                            "epoch": epoch,
                            "valid_equal_head_standardized_huber": best_metric,
                            "model_state": best_state,
                            "scalers": self.scalers,
                            "industry_vocab_size": self.industry_vocab_size,
                        },
                        output / "best.pt",
                    )
            if epoch >= self.min_epochs and stale >= self.patience:
                break
        if best_state is None:
            raise CourageStrictV1TrainingError("training produced no checkpoint")
        network.load_state_dict(best_state)
        self.model_state = copy.deepcopy(best_state)
        self.history = history
        self.fitted = True
        if evals_result is not None:
            evals_result["history"] = copy.deepcopy(history)
            evals_result["best_valid_metric"] = best_metric
        if world_size > 1:
            dist.barrier()

    def predict(
        self,
        dataset: CourageStrictV1Dataset,
        segment: str = "valid",
    ) -> pd.DataFrame:
        if not self.fitted or self.model_state is None or self.scalers is None:
            raise CourageStrictV1TrainingError("model is not fitted")
        if self.industry_vocab_size is None:
            raise CourageStrictV1TrainingError("industry vocabulary is absent")
        device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = PatchTSTCourageStrictV1(
            industry_vocab_size=self.industry_vocab_size
        ).to(device)
        model.load_state_dict(self.model_state)
        model.eval()
        data = dataset.prepare(segment, scalers=self.scalers, cache_capacity=8)
        loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=self.num_workers > 0,
        )
        predictions: list[np.ndarray] = []
        symbols: list[np.ndarray] = []
        calendar_indices: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                device_batch = _move(batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    scaled = _forward(model, device_batch).float().cpu().numpy()
                for position, horizon in enumerate(HORIZONS_V1):
                    stats = self.scalers["targets"][str(horizon)]
                    scaled[:, position] = (
                        scaled[:, position] * stats["iqr"] + stats["median"]
                    )
                predictions.append(scaled)
                symbols.append(batch["symbol_position"].numpy())
                calendar_indices.append(batch["calendar_index"].numpy())
        values = np.concatenate(predictions)
        symbol_positions = np.concatenate(symbols)
        times = np.concatenate(calendar_indices)
        provider = data.raw.provider
        index = pd.MultiIndex.from_arrays(
            [
                provider.calendar[times],
                np.asarray(provider.qlib_symbols)[symbol_positions],
            ],
            names=["datetime", "instrument"],
        )
        return pd.DataFrame(
            values,
            index=index,
            columns=[f"prediction_{horizon}" for horizon in HORIZONS_V1],
        )


__all__ = ["CourageStrictV1Model", "CourageStrictV1TrainingError"]
