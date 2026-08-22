# Courage Strict Continuous V1

This directory contains the bounded, continuous one-epoch PatchTST experiment
for the frozen `courage_strict_v1` provider.  The pushed scope is the base
5%--15% PIT experiment and its read-only baseline closure; generated data,
checkpoints, predictions, and later 10%--15% variants are intentionally absent.

## Read first

- training configuration: `config.json`;
- training entry point: `run.py`;
- full rolling-Valid evaluator: `evaluate_origin.py`;
- Train/Valid curve renderer: `plot_curves.py`（同时强制生成综合曲线和7个horizon独立Train/Valid曲线）；
- frozen baseline contract: `baseline_closure_contract_v1.json`;
- baseline closure evaluator: `evaluate_baseline_closure_v1.py`;
- result: `docs/courage_strict_continuous_v1/BASELINE_CLOSURE_EVALUATION.md`.

训练中的轻量指标按配置的validation step记录。日期、分钟位置、PIT行业和换手率等昂贵分组只在best checkpoint冻结后，由baseline closure evaluator对完整固定Valid执行一次；不允许用分组结果重新选择checkpoint。

## Model and data path

```text
CourageStrictV1Dataset
  -> 1200-minute dynamic sequence + T-1 slow features + industry id
  -> PatchTSTCourageStrictV1
  -> seven return heads: 5/15/30/60/120/240/480 minutes
```

Training uses eight-GPU BF16 DDP, global batch 1024, time-cell stratified
sampling, and full fixed-Valid checkpoint selection.  Runtime outputs belong
under `artifacts/` and are not committed.

## Current result

The frozen step-2250 checkpoint fails the completed baseline gate: mean RMSE
skill versus the strongest preregistered baseline is `-0.184%`, with `0/7`
positive horizons.  The 240/480-minute heads retain weak ranking information,
but absolute-return calibration fails.  No later origin should be promoted from
this result without completing the remaining sanity checks.
