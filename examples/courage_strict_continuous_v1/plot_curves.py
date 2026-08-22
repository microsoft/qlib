"""Render loss and optimization curves for one completed origin."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qlib.contrib.model.patchtst_courage_strict_v1 import HORIZONS_V1


def load_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    steps = [int(item["global_step"]) for item in records]
    if not records or any(left >= right for left, right in zip(steps, steps[1:])):
        raise ValueError("curve records must be non-empty and strictly increasing")
    return records


def render(origin_root: Path, *, doc_output: Path | None = None) -> None:
    records = load_records(origin_root / "loss_curve.json")
    loss_function = str(records[0].get("loss_function", "HUBER")).upper()
    suffix = loss_function.lower()
    train_key = f"train_window_equal_head_standardized_{suffix}"
    valid_key = f"valid_equal_head_standardized_{suffix}"
    train_per_head_key = f"train_window_per_head_standardized_{suffix}"
    valid_per_head_key = f"valid_per_head_standardized_{suffix}"
    if any(
        item.get("loss_function", "HUBER").upper() != loss_function for item in records
    ):
        raise ValueError("loss function changes inside one curve")
    steps = [item["global_step"] for item in records]
    train = [item[train_key] for item in records]
    valid = [item[valid_key] for item in records]
    best = min(records, key=lambda item: item[valid_key])

    figure, axis = plt.subplots(figsize=(11, 6))
    intervals = [right - left for left, right in zip(steps, steps[1:])]
    # The final partial window can be shorter than the configured cadence
    # (1590 is not divisible by 50), so label the modal interval rather than
    # the minimum tail interval.
    interval = (
        collections.Counter(intervals).most_common(1)[0][0] if intervals else steps[0]
    )
    axis.plot(
        steps,
        train,
        marker="o",
        markersize=3,
        label=f"Train {interval}-step window",
    )
    axis.plot(steps, valid, marker="o", markersize=3, label="Full Valid")
    axis.axvline(best["global_step"], color="tab:red", linestyle="--", label="Best")
    axis.set_xlabel("Global optimizer step")
    axis.set_ylabel(f"Equal-head standardized {loss_function}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    legacy_curve = origin_root / "train_valid_loss_curve.png"
    canonical_curve = origin_root / "TRAIN_VALID_LOSS_CURVE.png"
    figure.savefig(legacy_curve, dpi=180)
    shutil.copyfile(legacy_curve, canonical_curve)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 7))
    for horizon in HORIZONS_V1:
        axis.plot(
            steps,
            [item[valid_per_head_key][str(horizon)] for item in records],
            label=f"{horizon}m",
        )
    axis.axvline(best["global_step"], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Global optimizer step")
    axis.set_ylabel(f"Full Valid standardized {loss_function}")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(origin_root / "per_head_valid_loss_curve.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    flattened = axes.ravel()
    for axis, horizon in zip(flattened, HORIZONS_V1, strict=False):
        key = str(horizon)
        axis.plot(
            steps,
            [item[train_per_head_key][key] for item in records],
            marker="o",
            markersize=2,
            label=f"Train {interval}-step window",
        )
        axis.plot(
            steps,
            [item[valid_per_head_key][key] for item in records],
            marker="o",
            markersize=2,
            label="Full Valid",
        )
        axis.axvline(best["global_step"], color="tab:red", linestyle="--", linewidth=1)
        axis.set_title(f"{horizon}m")
        axis.set_ylabel(f"Standardized {loss_function}")
        axis.grid(alpha=0.25)
        axis.legend()
    flattened[-1].set_visible(False)
    for axis in flattened[-2:]:
        axis.set_xlabel("Global optimizer step")
    figure.suptitle("Per-horizon Train-window versus fixed-Valid loss", y=0.995)
    figure.tight_layout()
    per_head_train_valid = origin_root / "PER_HEAD_TRAIN_VALID_LOSS_CURVES.png"
    figure.savefig(per_head_train_valid, dpi=180)
    plt.close(figure)

    figure, left = plt.subplots(figsize=(11, 6))
    right = left.twinx()
    left.plot(
        steps,
        [item["gradient"]["median_pre_clip"] for item in records],
        label="Grad median",
    )
    left.plot(
        steps,
        [item["gradient"]["p95_pre_clip"] for item in records],
        label="Grad p95",
    )
    left.plot(
        steps,
        [item["gradient"]["clip_trigger_ratio"] for item in records],
        label="Clip ratio",
    )
    right.plot(
        steps,
        [item["learning_rate"] for item in records],
        color="black",
        linestyle="--",
        label="Learning rate",
    )
    left.set_xlabel("Global optimizer step")
    left.set_ylabel("Gradient norm / ratio")
    right.set_ylabel("Learning rate")
    left.grid(alpha=0.25)
    handles, labels = left.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    left.legend(handles + handles2, labels + labels2, ncol=2)
    figure.tight_layout()
    figure.savefig(origin_root / "gradient_lr_curve.png", dpi=180)
    plt.close(figure)
    if doc_output is not None:
        doc_output.mkdir(parents=True, exist_ok=True)
        for source in (canonical_curve, per_head_train_valid):
            destination = doc_output / source.name
            shutil.copyfile(source, destination)
            if destination.read_bytes() != source.read_bytes():
                raise ValueError("documentation curve copy drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin_root", type=Path)
    parser.add_argument("--doc-output", type=Path)
    args = parser.parse_args()
    render(
        args.origin_root.resolve(),
        doc_output=args.doc_output.resolve() if args.doc_output else None,
    )


if __name__ == "__main__":
    main()
