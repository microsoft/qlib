"""Render the original-V1 step diagnostic closure report and figure."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from examples.courage_strict_continuous_v1.plot_curves import render as render_curves
from qlib.contrib.model.courage_strict_continuous_v1 import sha256_file
from qlib.contrib.model.patchtst_courage_strict_v1 import HORIZONS_V1

ROOT = Path(__file__).resolve().parents[2]
RUN = (
    ROOT / "artifacts/courage_strict_continuous_v1_diagnostic_v1/run_v1/origin_2026_01"
)
BASELINE = (
    ROOT / "artifacts/courage_strict_continuous_v1_diagnostic_v1/baseline_closure_v1"
)
DOC = ROOT / "docs/courage_strict_continuous_v1_diagnostic_v1"
ORIGINAL = ROOT / "artifacts/courage_strict_continuous_v1/run_v1/origin_2026_01"
GROUPED_DIMENSIONS = {"date", "minute_slot", "industry_id", "turnover_band"}


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostic_figure(records: list[dict], output: Path) -> None:
    steps = [int(row["global_step"]) for row in records]
    figure, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for horizon in HORIZONS_V1:
        key = str(horizon)
        values = [row["valid_prediction_diagnostics"][key] for row in records]
        axes[0].plot(
            steps,
            [row["prediction_std"] / row["target_std"] for row in values],
            label=f"{horizon}m",
        )
        axes[1].plot(
            steps,
            [row["prediction_up_rate"] for row in values],
            label=f"{horizon}m",
        )
        axes[2].plot(
            steps,
            [row["bias"] / row["target_std"] for row in values],
            label=f"{horizon}m",
        )
    best = min(records, key=lambda row: row["valid_equal_head_standardized_huber"])
    for axis in axes:
        axis.axvline(best["global_step"], color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Prediction std / target std")
    axes[1].set_ylabel("Prediction up rate")
    axes[2].set_ylabel("Bias / target std")
    axes[2].set_xlabel("Global optimizer step")
    axes[0].legend(ncol=4)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _validate_grouped_metrics(path: Path, baseline_manifest: dict) -> pd.DataFrame:
    recorded = baseline_manifest.get("outputs", {}).get("grouped_model_metrics.csv")
    if (
        not path.is_file()
        or not isinstance(recorded, dict)
        or sha256_file(path) != recorded.get("sha256")
    ):
        raise RuntimeError("best-checkpoint grouped metrics identity drift")
    frame = pd.read_csv(path)
    required_columns = {
        "dimension",
        "group",
        "horizon",
        "rows",
        "rmse",
        "bias",
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "auc",
        "target_up_rate",
        "prediction_up_rate",
    }
    if not required_columns.issubset(frame.columns):
        raise RuntimeError("best-checkpoint grouped metric columns incomplete")
    if set(frame["dimension"]) != GROUPED_DIMENSIONS:
        raise RuntimeError("best-checkpoint grouped dimensions incomplete")
    expected_horizons = set(HORIZONS_V1)
    for dimension in GROUPED_DIMENSIONS:
        observed = set(
            frame.loc[frame["dimension"].eq(dimension), "horizon"].astype(int)
        )
        if observed != expected_horizons:
            raise RuntimeError(f"grouped horizon coverage incomplete: {dimension}")
    return frame


def render() -> Path:
    records = _read(RUN / "loss_curve.json")
    manifest = _read(RUN / "_training_manifest.json")
    baseline = _read(BASELINE / "_manifest.json")
    original = _read(ORIGINAL / "_training_manifest.json")
    if not isinstance(records, list) or len(records) != 29:
        raise RuntimeError("diagnostic loss-curve identity drift")
    required = {
        "train_window_prediction_diagnostics",
        "valid_prediction_diagnostics",
    }
    if any(not required.issubset(row) for row in records):
        raise RuntimeError("step diagnostic fields are incomplete")
    if (
        manifest.get("decision") != "PASS_CONTINUOUS_ORIGIN_TRAINING_COMPLETE"
        or baseline.get("decision") != "FAIL_BASELINE_GATE"
        or manifest.get("june_or_later_read") is not False
        or baseline.get("april_or_later_read") is not False
    ):
        raise RuntimeError("diagnostic terminal boundary drift")
    DOC.mkdir(parents=True, exist_ok=True)
    render_curves(RUN, doc_output=DOC)
    grouped_source = BASELINE / "grouped_model_metrics.csv"
    grouped = _validate_grouped_metrics(grouped_source, baseline)
    grouped_document = DOC / "BEST_CHECKPOINT_GROUPED_METRICS.csv"
    shutil.copyfile(grouped_source, grouped_document)
    if grouped_document.read_bytes() != grouped_source.read_bytes():
        raise RuntimeError("grouped diagnostic documentation copy drift")
    diagnostic_figure = DOC / "PREDICTION_DIAGNOSTIC_CURVES.png"
    _diagnostic_figure(records, diagnostic_figure)
    best = min(records, key=lambda row: row["valid_equal_head_standardized_huber"])
    selected = {
        int(row["global_step"]): row
        for row in records
        if int(row["global_step"]) in {250, int(best["global_step"]), 7131}
    }
    lines = [
        "# Courage Strict Continuous V1 Diagnostic 复训报告",
        "",
        "## 结论",
        "",
        f"- 原V1人口和参数复训完成：Train `{manifest['train_rows']:,}`、Valid `{manifest['valid_rows']:,}`、`{manifest['steps_executed']}` steps；最佳仍为 step `{best['global_step']}`。",
        f"- 最佳Valid Huber `{best['valid_equal_head_standardized_huber']:.9f}`；原V1为 `{original['best_valid_equal_head_standardized_huber']:.9f}`，差值 `{best['valid_equal_head_standardized_huber'] - original['best_valid_equal_head_standardized_huber']:+.9f}`。",
        f"- 完整基线门：`{baseline['decision']}`；平均RMSE skill `{baseline['gate']['mean_rmse_skill_vs_best_baseline']:+.3%}`，正skill `{baseline['gate']['positive_horizons']}/7`。",
        "- 逐step日志确认输出收缩在训练早期已经形成；它不是最终checkpoint事后统计造成的假象。",
        "- 按预注册门禁停止，不训练Origin-2/3，不读取April或更晚数据。",
        "",
        "## Train / Valid loss",
        "",
        "![Train/Valid loss](TRAIN_VALID_LOSS_CURVE.png)",
        "",
        "## 逐horizon Train / Valid loss",
        "",
        "![Per-head Train/Valid loss](PER_HEAD_TRAIN_VALID_LOSS_CURVES.png)",
        "",
        "## Prediction动态诊断",
        "",
        "![Prediction diagnostics](PREDICTION_DIAGNOSTIC_CURVES.png)",
        "",
        "下表均为完整固定January Valid的原始收益尺度统计。`Std ratio`越接近0，输出越接近常数。",
        "",
        "| H | Target up | Step250 std/up/bias | Step2250 std/up/bias | Step7131 std/up/bias |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS_V1:
        key = str(horizon)
        values = [
            selected[step]["valid_prediction_diagnostics"][key]
            for step in (250, 2250, 7131)
        ]

        def cell(row: dict) -> str:
            return (
                f"{row['prediction_std'] / row['target_std']:.3f} / "
                f"{row['prediction_up_rate']:.1%} / {row['bias']:+.3%}"
            )

        lines.append(
            f"| {horizon} | {values[0]['target_up_rate']:.1%} | {cell(values[0])} | "
            f"{cell(values[1])} | {cell(values[2])} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- 5m/15m很快退化为近乎永远预测下跌；最佳step时上涨率仅约0.1%/0.4%。",
        "- 30m—120m虽然保留少量方向变化，但prediction std仍远低于target std。",
        "- 240m/480m具有更高AUC和Rank IC，但绝对收益RMSE仍输给zero/Train mean；排序信息不能替代条件均值校准。",
        "- Train loss继续下降而Valid在step2250后不再刷新，说明继续增加epoch不是当前修复方向。",
        "",
        "## 证据",
        "",
        "- [完整基线、逐日和分组报告](BASELINE_CLOSURE_EVALUATION.md)",
        f"- [最佳checkpoint分组指标](BEST_CHECKPOINT_GROUPED_METRICS.csv)：`{len(grouped):,}`行，覆盖日期、分钟位置、PIT行业和换手率。",
        "- [既有三项Sanity Check](../courage_strict_continuous_v1/SANITY_CHECK_REPORT.md)",
        "- [既有10-seed shuffled-label复核](../courage_strict_continuous_v1/SANITY_NULL_DISTRIBUTION_REPORT.md)",
        "- [既有January—March冻结迁移诊断](../courage_strict_continuous_v1/FROZEN_ORIGIN1_TEMPORAL_TRANSFER_EVALUATION.md)",
        "",
        "## 边界",
        "",
        "- 本次没有改变原V1 provider、PIT人口、Feature、Label、模型、loss、seed或优化参数。",
        "- 新checkpoint仅属于diagnostic命名空间，不覆盖、不晋升原V1 checkpoint。",
        "- 未读取2026-02-02及以后数据；未执行refit、策略、回测、交易或远端推送。",
        "",
    ]
    report = DOC / "TRAINING_AND_DIAGNOSTIC_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    shutil.copyfile(
        BASELINE / "BASELINE_CLOSURE_EVALUATION.md",
        DOC / "BASELINE_CLOSURE_EVALUATION.md",
    )
    output_manifest = {
        "schema_version": "courage_strict_continuous_diagnostic_report_manifest_v1",
        "decision": baseline["decision"],
        "best_step": int(best["global_step"]),
        "training_manifest_sha256": sha256_file(RUN / "_training_manifest.json"),
        "loss_curve_sha256": sha256_file(RUN / "loss_curve.json"),
        "train_valid_curve_sha256": sha256_file(DOC / "TRAIN_VALID_LOSS_CURVE.png"),
        "per_head_train_valid_curve_sha256": sha256_file(
            DOC / "PER_HEAD_TRAIN_VALID_LOSS_CURVES.png"
        ),
        "prediction_diagnostic_curve_sha256": sha256_file(diagnostic_figure),
        "best_checkpoint_grouped_metrics_sha256": sha256_file(grouped_document),
        "best_checkpoint_grouped_metric_rows": len(grouped),
        "best_checkpoint_grouped_dimensions": sorted(GROUPED_DIMENSIONS),
        "baseline_manifest_sha256": sha256_file(BASELINE / "_manifest.json"),
        "report_sha256": sha256_file(report),
        "april_or_later_read": False,
        "checkpoint_promoted": False,
    }
    (DOC / "_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(render())
