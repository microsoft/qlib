"""Read-only Qlib and model-dataset acceptance check for Courage Strict V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.courage_strict_v1.dataset import CourageStrictV1Dataset
from qlib.data import D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qlib-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    root = args.qlib_root.resolve()
    provider = root / "data/courage_strict_v1/qlib_provider"
    catalog = json.loads(
        (provider / "_courage_strict_v1_qlib_catalog.json").read_text(encoding="utf-8")
    )
    if catalog.get("decision") != "PASS_V1_QLIB_MATERIALIZATION_AND_PARITY":
        raise RuntimeError("provider terminal decision is not PASS")
    qlib.init(provider_uri={"1min": str(provider)}, region=REG_CN)
    instrument = (provider / "instruments/all.txt").read_text().split("\t", 1)[0]
    frame = D.features(
        [instrument],
        ["$open", "$close", "$volume", "$stock_ret_1", "$label_return_5"],
        start_time="2025-07-01 09:32:00",
        end_time="2025-07-01 10:00:00",
        freq="1min",
    )
    if frame.empty or not np.isfinite(frame["$open"].dropna()).all():
        raise RuntimeError("Qlib standard feature read failed")
    dataset = CourageStrictV1Dataset(
        provider_root=provider,
        segments={
            "train": ("2025-07-01", "2026-03-02"),
            "valid": ("2026-03-02", "2026-04-01"),
        },
    )
    train = dataset.prepare("train", cache_capacity=1)
    valid = dataset.prepare("valid", cache_capacity=1)
    item = train[0]
    if item["values"].shape != (1200, 12) or item["targets"].shape != (7,):
        raise RuntimeError("model sample geometry drift")
    print(
        json.dumps(
            {
                "decision": "PASS_V1_QLIB_PROVIDER_AND_DATASET_READ",
                "provider_coverage": catalog["coverage"],
                "train_samples": len(train),
                "valid_samples": len(valid),
                "qlib_probe_rows": len(frame),
                "first_sample_shapes": {
                    key: list(value.shape)
                    for key, value in item.items()
                    if hasattr(value, "shape")
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
