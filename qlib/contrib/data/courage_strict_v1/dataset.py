"""Qlib-native map dataset and Train-only scaling for Courage Strict V1."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from qlib.contrib.data.courage_strict_v1.minute_courage_strict_c4_features_v1 import (
    DYNAMIC_FEATURES,
    SLOW_FEATURES,
)
from qlib.contrib.model.patchtst_courage_strict_v1 import HORIZONS_V1, LOOKBACK_V1
from qlib.data.dataset import Dataset


class CourageStrictV1DatasetError(RuntimeError):
    """Raised when the V1 provider cannot form an exact model sample."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def scaler_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _from_qlib(symbol: str) -> str:
    if len(symbol) != 8 or symbol[:2] not in {"SH", "SZ"}:
        raise CourageStrictV1DatasetError(f"invalid Qlib instrument: {symbol}")
    return f"{symbol[2:]}.{symbol[:2]}"


class _Provider:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        catalog_path = self.root / "_courage_strict_v1_qlib_catalog.json"
        if not catalog_path.is_file():
            raise CourageStrictV1DatasetError("provider terminal catalog is absent")
        self.catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        accepted_decisions = {
            "PASS_V1_QLIB_MATERIALIZATION_AND_PARITY",
            "PASS_V1_APRIL_REPLAY_PROVIDER_MATERIALIZATION",
            "PASS_SOURCE_SEMANTICS_V2_FEATURE_SEQUENCE_MATERIALIZATION",
        }
        if self.catalog.get("decision") not in accepted_decisions:
            raise CourageStrictV1DatasetError("provider is not accepted")
        coverage = self.catalog.get("coverage", {})
        expected_rows = int(coverage.get("calendar_rows", 0))
        expected_instruments = int(coverage.get("instruments", 0))
        if expected_rows <= LOOKBACK_V1 or expected_instruments <= 0:
            raise CourageStrictV1DatasetError("provider catalog calendar drift")
        calendar_path = self.root / "calendars/1min.txt"
        self.calendar = pd.DatetimeIndex(
            pd.to_datetime(calendar_path.read_text(encoding="utf-8").splitlines())
        )
        if (
            len(self.calendar) != expected_rows
            or not self.calendar.is_monotonic_increasing
        ):
            raise CourageStrictV1DatasetError("provider calendar drift")
        lines = (
            (self.root / "instruments/all.txt").read_text(encoding="utf-8").splitlines()
        )
        self.qlib_symbols = tuple(line.split("\t", 1)[0] for line in lines if line)
        self.symbols = tuple(_from_qlib(symbol) for symbol in self.qlib_symbols)
        if len(self.symbols) != expected_instruments:
            raise CourageStrictV1DatasetError("provider instrument coverage drift")

    def path(self, symbol_position: int, field: str) -> Path:
        qlib_symbol = self.qlib_symbols[int(symbol_position)].lower()
        return self.root / "features" / qlib_symbol / f"{field.lower()}.1min.bin"

    def array(self, symbol_position: int, field: str) -> np.memmap:
        path = self.path(symbol_position, field)
        expected_bytes = (len(self.calendar) + 1) * 4
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise CourageStrictV1DatasetError(f"Qlib bin drift: {path}")
        return np.memmap(
            path, dtype="<f4", mode="r", offset=4, shape=(len(self.calendar),)
        )


def _build_references(
    provider: _Provider,
    *,
    start: str,
    end: str,
    signal_stride: int = 5,
    turnover_band: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if signal_stride not in {1, 5}:
        raise CourageStrictV1DatasetError("supported signal strides are 1 and 5")
    segment = (provider.calendar >= pd.Timestamp(start)) & (
        provider.calendar < pd.Timestamp(end)
    )
    symbol_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    for position in range(len(provider.symbols)):
        member = provider.array(position, "pit_member") > 0.5
        if turnover_band is not None:
            lower, upper = map(float, turnover_band)
            if not 5.0 <= lower < upper <= 15.0:
                raise CourageStrictV1DatasetError("turnover sub-band is invalid")
            turnover = provider.array(position, "turnover_mean_60")
            member &= np.isfinite(turnover) & (turnover >= lower) & (turnover <= upper)
        legal = provider.array(position, "signal_legal") > 0.5
        stride = (
            provider.array(position, "sample_present") > 0.5
            if signal_stride == 1
            else provider.array(position, "signal_stride5") > 0.5
        )
        indices = np.flatnonzero(segment & member & legal & stride)
        indices = indices[indices >= LOOKBACK_V1 - 1]
        if len(indices):
            symbol_parts.append(np.full(len(indices), position, dtype=np.int32))
            end_parts.append(indices.astype(np.int32, copy=False))
    if not end_parts:
        raise CourageStrictV1DatasetError("segment has no samples")
    return np.concatenate(symbol_parts), np.concatenate(end_parts)


def label_maturity_mask_v1(
    valid: np.ndarray,
    target_end: np.ndarray,
    available: np.ndarray,
    *,
    cutoff_index: int,
) -> np.ndarray:
    """Purge one segment/head by both Label end and availability index."""
    return (
        np.asarray(valid, dtype=bool)
        & np.isfinite(target_end)
        & np.isfinite(available)
        & (np.asarray(target_end) < int(cutoff_index))
        & (np.asarray(available) < int(cutoff_index))
    )


class CourageStrictV1TorchDataset(TorchDataset[dict[str, torch.Tensor]]):
    """Map-style dataset backed only by standard Qlib 1-minute bins."""

    def __init__(
        self,
        *,
        provider_root: Path,
        start: str,
        end: str,
        cache_capacity: int = 8,
        signal_stride: int = 5,
        turnover_band: tuple[float, float] | None = None,
    ) -> None:
        if cache_capacity <= 0:
            raise CourageStrictV1DatasetError("cache capacity must be positive")
        self.provider = _Provider(provider_root)
        self.symbol_positions, self.ends = _build_references(
            self.provider,
            start=start,
            end=end,
            signal_stride=signal_stride,
            turnover_band=turnover_band,
        )
        self.start, self.end = start, end
        self.label_cutoff_index = int(
            self.provider.calendar.searchsorted(pd.Timestamp(end), side="left")
        )
        self.cache_capacity = int(cache_capacity)
        self.signal_stride = int(signal_stride)
        self.turnover_band = turnover_band
        self._cache: collections.OrderedDict[int, dict[str, np.memmap]] = (
            collections.OrderedDict()
        )

    def __len__(self) -> int:
        return len(self.ends)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = collections.OrderedDict()
        return state

    def _arrays(self, symbol_position: int) -> dict[str, np.memmap]:
        arrays = self._cache.pop(symbol_position, None)
        if arrays is None:
            fields = [
                "minute_slot",
                "session_id",
                "industry_id",
                "industry_known",
            ]
            fields += [
                name
                for feature in DYNAMIC_FEATURES
                for name in (
                    feature,
                    f"{feature}__available",
                    f"{feature}__data_missing",
                )
            ]
            fields += [
                name
                for feature in SLOW_FEATURES
                for name in (
                    feature,
                    f"{feature}__available",
                    f"{feature}__data_missing",
                )
            ]
            fields += [
                name
                for horizon in HORIZONS_V1
                for name in (
                    f"label_return_{horizon}",
                    f"label_valid_{horizon}",
                    f"label_target_end_index_{horizon}",
                    f"label_available_index_{horizon}",
                )
            ]
            arrays = {
                field: self.provider.array(symbol_position, field) for field in fields
            }
        self._cache[symbol_position] = arrays
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        symbol_position = int(self.symbol_positions[int(item)])
        end = int(self.ends[int(item)])
        start = end - LOOKBACK_V1 + 1
        arrays = self._arrays(symbol_position)
        values = np.column_stack(
            [arrays[feature][start : end + 1] for feature in DYNAMIC_FEATURES]
        ).astype(np.float32, copy=False)
        available = np.column_stack(
            [
                arrays[f"{feature}__available"][start : end + 1] > 0.5
                for feature in DYNAMIC_FEATURES
            ]
        )
        missing = np.column_stack(
            [
                arrays[f"{feature}__data_missing"][start : end + 1] > 0.5
                for feature in DYNAMIC_FEATURES
            ]
        )
        usable = available & ~missing & np.isfinite(values)
        values = np.where(usable, values, 0.0).astype(np.float32, copy=False)
        slow_values = np.array(
            [arrays[feature][end] for feature in SLOW_FEATURES], dtype=np.float32
        )
        slow_available = np.array(
            [
                arrays[f"{feature}__available"][end] > 0.5
                and arrays[f"{feature}__data_missing"][end] <= 0.5
                for feature in SLOW_FEATURES
            ],
            dtype=bool,
        )
        slow_values[~slow_available] = 0.0
        targets = np.array(
            [arrays[f"label_return_{horizon}"][end] for horizon in HORIZONS_V1],
            dtype=np.float32,
        )
        target_mask = np.array(
            [arrays[f"label_valid_{horizon}"][end] > 0.5 for horizon in HORIZONS_V1],
            dtype=bool,
        )
        label_end = np.array(
            [
                arrays[f"label_target_end_index_{horizon}"][end]
                for horizon in HORIZONS_V1
            ],
            dtype=np.float64,
        )
        label_available = np.array(
            [
                arrays[f"label_available_index_{horizon}"][end]
                for horizon in HORIZONS_V1
            ],
            dtype=np.float64,
        )
        target_mask = label_maturity_mask_v1(
            target_mask,
            label_end,
            label_available,
            cutoff_index=self.label_cutoff_index,
        )
        if not np.isfinite(targets[target_mask]).all():
            raise CourageStrictV1DatasetError("valid target is not finite")
        targets[~target_mask] = 0.0
        minute_index = arrays["minute_slot"][start : end + 1].astype(np.int64)
        session_index = arrays["session_id"][start : end + 1].astype(np.int64)
        return {
            "values": torch.from_numpy(values.copy()),
            "available": torch.from_numpy(available.copy()),
            "missing": torch.from_numpy(missing.copy()),
            "padding": torch.zeros(LOOKBACK_V1, dtype=torch.bool),
            "minute_index": torch.from_numpy(minute_index.copy()),
            "session_index": torch.from_numpy(session_index.copy()),
            "slow_values": torch.from_numpy(slow_values),
            "slow_available": torch.from_numpy(slow_available),
            "industry_id": torch.tensor(
                int(arrays["industry_id"][end]), dtype=torch.long
            ),
            "targets": torch.from_numpy(targets.copy()),
            "raw_targets": torch.from_numpy(targets.copy()),
            "target_mask": torch.from_numpy(target_mask),
            "symbol_position": torch.tensor(symbol_position, dtype=torch.int32),
            "calendar_index": torch.tensor(end, dtype=torch.int32),
        }


def _merged_intervals(ends: np.ndarray) -> list[tuple[int, int]]:
    unique = np.unique(np.asarray(ends, dtype=np.int64))
    if len(unique) == 0:
        return []
    starts = unique - LOOKBACK_V1 + 1
    output: list[tuple[int, int]] = []
    start, stop = int(starts[0]), int(unique[0] + 1)
    for next_start, next_stop in zip(starts[1:], unique[1:] + 1, strict=True):
        if int(next_start) <= stop:
            stop = max(stop, int(next_stop))
        else:
            output.append((start, stop))
            start, stop = int(next_start), int(next_stop)
    output.append((start, stop))
    return output


def _stats(values: np.ndarray, *, clip: bool) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise CourageStrictV1DatasetError("scaler input is empty")
    q = np.quantile(finite, [0.001, 0.25, 0.5, 0.75, 0.999], method="linear")
    iqr = float(q[3] - q[1])
    if not np.isfinite(iqr) or iqr <= 1e-8:
        raise CourageStrictV1DatasetError("scaler IQR is degenerate")
    return {
        "count": len(finite),
        "lower": float(q[0]) if clip else None,
        "median": float(q[2]),
        "iqr": iqr,
        "upper": float(q[4]) if clip else None,
        "clip": clip,
    }


def fit_train_scalers(dataset: CourageStrictV1TorchDataset) -> dict[str, Any]:
    dynamic_chunks = {feature: [] for feature in DYNAMIC_FEATURES}
    slow_chunks = {feature: [] for feature in SLOW_FEATURES}
    target_chunks = {horizon: [] for horizon in HORIZONS_V1}
    for position in np.unique(dataset.symbol_positions):
        selected = dataset.symbol_positions == position
        ends = dataset.ends[selected]
        arrays = dataset._arrays(int(position))
        for start, stop in _merged_intervals(ends):
            for feature in DYNAMIC_FEATURES:
                values = np.asarray(arrays[feature][start:stop])
                usable = (
                    (arrays[f"{feature}__available"][start:stop] > 0.5)
                    & (arrays[f"{feature}__data_missing"][start:stop] <= 0.5)
                    & np.isfinite(values)
                )
                if usable.any():
                    dynamic_chunks[feature].append(values[usable].astype(np.float64))
        for feature in SLOW_FEATURES:
            values = np.asarray(arrays[feature][ends])
            usable = (
                (arrays[f"{feature}__available"][ends] > 0.5)
                & (arrays[f"{feature}__data_missing"][ends] <= 0.5)
                & np.isfinite(values)
            )
            if usable.any():
                slow_chunks[feature].append(values[usable].astype(np.float64))
        for horizon in HORIZONS_V1:
            values = np.asarray(arrays[f"label_return_{horizon}"][ends])
            target_end = arrays[f"label_target_end_index_{horizon}"][ends]
            available = arrays[f"label_available_index_{horizon}"][ends]
            usable = label_maturity_mask_v1(
                (arrays[f"label_valid_{horizon}"][ends] > 0.5) & np.isfinite(values),
                target_end,
                available,
                cutoff_index=dataset.label_cutoff_index,
            )
            if usable.any():
                target_chunks[horizon].append(values[usable].astype(np.float64))
    result = {
        "schema_version": "courage_strict_v1_train_scalers_v1",
        "fit_role": "train_only",
        "provider_catalog_sha256": _sha256_file(
            dataset.provider.root / "_courage_strict_v1_qlib_catalog.json"
        ),
        "train_segment": [dataset.start, dataset.end],
        "train_samples": len(dataset),
        "train_reference_sha256": hashlib.sha256(
            dataset.symbol_positions.tobytes(order="C")
            + dataset.ends.tobytes(order="C")
        ).hexdigest(),
        "dynamic_order": list(DYNAMIC_FEATURES),
        "slow_order": list(SLOW_FEATURES),
        "horizons": list(HORIZONS_V1),
        "dynamic": {
            feature: _stats(np.concatenate(chunks), clip=True)
            for feature, chunks in dynamic_chunks.items()
        },
        "slow": {
            feature: _stats(np.concatenate(chunks), clip=True)
            for feature, chunks in slow_chunks.items()
        },
        "targets": {
            str(horizon): _stats(np.concatenate(chunks), clip=False)
            for horizon, chunks in target_chunks.items()
        },
    }
    result["scaler_sha256"] = scaler_sha256(result)
    return result


def _scale(
    values: torch.Tensor,
    available: torch.Tensor,
    stats: list[dict[str, Any]],
) -> None:
    for index, item in enumerate(stats):
        usable = available[..., index]
        if item["clip"]:
            values[..., index].clamp_(float(item["lower"]), float(item["upper"]))
        values[..., index].sub_(float(item["median"])).div_(float(item["iqr"]))
        values[..., index].masked_fill_(~usable, 0.0)


class CourageStrictV1ScaledDataset(TorchDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        raw: CourageStrictV1TorchDataset,
        scalers: dict[str, Any],
    ) -> None:
        if scalers.get("fit_role") != "train_only":
            raise CourageStrictV1DatasetError("non-Train scaler rejected")
        catalog_sha = _sha256_file(
            raw.provider.root / "_courage_strict_v1_qlib_catalog.json"
        )
        if scalers.get("provider_catalog_sha256") != catalog_sha:
            raise CourageStrictV1DatasetError("scaler provider identity drift")
        self.raw, self.scalers = raw, scalers

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.raw[index]
        values = item["values"].clone()
        _scale(
            values,
            item["available"] & ~item["missing"],
            [self.scalers["dynamic"][feature] for feature in DYNAMIC_FEATURES],
        )
        slow_values = item["slow_values"].clone()
        _scale(
            slow_values.unsqueeze(0),
            item["slow_available"].unsqueeze(0),
            [self.scalers["slow"][feature] for feature in SLOW_FEATURES],
        )
        targets = item["targets"].clone()
        for position, horizon in enumerate(HORIZONS_V1):
            if item["target_mask"][position]:
                stats = self.scalers["targets"][str(horizon)]
                targets[position] = (targets[position] - stats["median"]) / stats["iqr"]
            else:
                targets[position] = 0.0
        return item | {"values": values, "slow_values": slow_values, "targets": targets}


class CourageStrictV1Dataset(Dataset):
    """Qlib Dataset facade returning model-ready Torch datasets."""

    def __init__(
        self,
        *,
        provider_root: str | Path,
        segments: dict[str, tuple[str, str]],
        signal_strides: dict[str, int] | None = None,
        turnover_band: tuple[float, float] | None = None,
    ) -> None:
        self.provider_root = Path(provider_root)
        self.segments = dict(segments)
        self.signal_strides = dict(signal_strides or {})
        self.turnover_band = turnover_band
        super().__init__()

    def prepare(
        self,
        segment: str,
        *,
        scalers: dict[str, Any] | None = None,
        cache_capacity: int = 8,
    ) -> TorchDataset:
        if segment not in self.segments:
            raise CourageStrictV1DatasetError(f"unknown segment: {segment}")
        start, end = self.segments[segment]
        raw = CourageStrictV1TorchDataset(
            provider_root=self.provider_root,
            start=start,
            end=end,
            cache_capacity=cache_capacity,
            signal_stride=int(self.signal_strides.get(segment, 5)),
            turnover_band=self.turnover_band,
        )
        return (
            raw
            if scalers is None
            else CourageStrictV1ScaledDataset(raw=raw, scalers=scalers)
        )


__all__ = [
    "CourageStrictV1Dataset",
    "CourageStrictV1DatasetError",
    "CourageStrictV1ScaledDataset",
    "CourageStrictV1TorchDataset",
    "fit_train_scalers",
    "label_maturity_mask_v1",
    "scaler_sha256",
]
