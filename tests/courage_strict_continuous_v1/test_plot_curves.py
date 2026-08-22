from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from examples.courage_strict_continuous_v1.plot_curves import load_records, render
from examples.courage_strict_continuous_v1.render_diagnostic_report_v1 import (
    GROUPED_DIMENSIONS,
    _validate_grouped_metrics,
)
from qlib.contrib.model.patchtst_courage_strict_v1 import HORIZONS_V1


def test_curve_loader_accepts_increasing_steps(tmp_path: Path) -> None:
    path = tmp_path / "curve.json"
    path.write_text(
        json.dumps([{"global_step": 250}, {"global_step": 500}]),
        encoding="utf-8",
    )
    assert [item["global_step"] for item in load_records(path)] == [250, 500]


def test_render_writes_canonical_run_and_document_curve(tmp_path: Path) -> None:
    records = []
    for step, train, valid in ((50, 0.7, 0.8), (100, 0.6, 0.75)):
        records.append(
            {
                "global_step": step,
                "train_window_equal_head_standardized_huber": train,
                "valid_equal_head_standardized_huber": valid,
                "valid_per_head_standardized_huber": {
                    str(horizon): valid for horizon in HORIZONS_V1
                },
                "train_window_per_head_standardized_huber": {
                    str(horizon): train for horizon in HORIZONS_V1
                },
                "gradient": {
                    "median_pre_clip": 0.2,
                    "p95_pre_clip": 0.3,
                    "clip_trigger_ratio": 0.0,
                },
                "learning_rate": 1e-4,
            }
        )
    (tmp_path / "loss_curve.json").write_text(json.dumps(records), encoding="utf-8")
    docs = tmp_path / "docs"

    render(tmp_path, doc_output=docs)

    canonical = tmp_path / "TRAIN_VALID_LOSS_CURVE.png"
    per_head = tmp_path / "PER_HEAD_TRAIN_VALID_LOSS_CURVES.png"
    assert canonical.is_file()
    assert canonical.read_bytes() == (docs / canonical.name).read_bytes()
    assert per_head.is_file()
    assert per_head.read_bytes() == (docs / per_head.name).read_bytes()


def test_best_checkpoint_grouped_metrics_cover_all_dimensions_and_heads(
    tmp_path: Path,
) -> None:
    rows = []
    for dimension in sorted(GROUPED_DIMENSIONS):
        for horizon in HORIZONS_V1:
            rows.append(
                {
                    "dimension": dimension,
                    "group": "sample",
                    "horizon": horizon,
                    "rows": 100,
                    "rmse": 0.1,
                    "bias": 0.0,
                    "accuracy": 0.5,
                    "balanced_accuracy": 0.5,
                    "mcc": 0.0,
                    "auc": 0.5,
                    "target_up_rate": 0.5,
                    "prediction_up_rate": 0.5,
                }
            )
    path = tmp_path / "grouped_model_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"outputs": {"grouped_model_metrics.csv": {"sha256": digest}}}
    observed = _validate_grouped_metrics(path, manifest)
    assert len(observed) == len(GROUPED_DIMENSIONS) * len(HORIZONS_V1)
