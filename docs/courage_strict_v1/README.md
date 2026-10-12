# Courage Strict V1

Status: `IMPLEMENTATION_IN_PROGRESS / QLIB_NATIVE / NO_OLD_LEARNED_STATE`

This directory documents the new `courage_strict_v1`, implemented directly in the Rolexl Qlib fork. Historical
evidence and engineering decision records are preserved under `evidence/`; they are provenance, not a dependency
on another project.

The implementation constructs the official-minute/PIT/label/feature data chain and materializes its own Qlib local
provider before fitting a new PatchTST model from random initialization. It does not import checkpoints, fitted
scalers, optimizer state, predictions, or historical training/evaluation artifacts.

## Frozen V1 semantics

- PIT pool: strict T-1, 60 completed sessions, at least 50 observations, arithmetic mean turnover in `[5%,15%]`;
- label: VWAP1 research gross return at 5/15/30/60/120/240/480 official minutes;
- inputs: 12 minute dynamic features, 5 strict-T-1 slow values, effective-dated industry embedding;
- sequence: 1200 official minutes, patch 30, stride 15;
- model: shared PatchTST, seven scalar return heads;
- loss/selection: equal-active-head standardized Huber (`delta=1.0`), Valid only;
- diagnostics include raw MAE/RMSE, bias, Pearson, Rank IC, ACC, balanced ACC, MCC and daily stability.

April 2026 is a framework-parity Development replay, not a new blind test. May and June are not inputs to this
V1; June remains sealed.

## Implementation stages

1. verify the preserved evidence and import the minimum canonical source facts into this project;
2. construct official axis, status, daily, industry, corporate-action and PIT membership state;
3. construct labels and V1 features directly into the `courage_strict_v1` Qlib provider;
4. run independent golden-period and Qlib-native parity checks;
5. profile Qlib loading/BF16/checkpoint-resume;
6. train from scratch, select on Valid and publish Development replay diagnostics.

## Canonical paths

- repository and command root: `/data1/lxl/workspace/datasets/.tmp/qlib`;
- virtual environment: `/data1/lxl/workspace/datasets/.tmp/qlib/.venv`;
- source facts: `/data1/lxl/workspace/datasets/.tmp/qlib/data/courage_strict_v1/source`;
- generated provider: `/data1/lxl/workspace/datasets/.tmp/qlib/data/courage_strict_v1/qlib_provider`;
- artifacts: `/data1/lxl/workspace/datasets/.tmp/qlib/artifacts/courage_strict_v1`.

All commands are run from the repository root. No active configuration or catalog resolves another project root.

## Commands

```bash
# Construct the governed PIT, Label, Feature and Qlib provider data.
.venv/bin/python examples/courage_strict_v1/build_data.py

# Verify standard Qlib expressions and the 1200×12 model sample interface.
.venv/bin/python examples/courage_strict_v1/verify_provider.py

# Formal eight-GPU BF16/DDP training, Valid selection and Qlib Recorder output.
examples/courage_strict_v1/launch_training.sh
```

The formal training command fails closed when CUDA is unavailable. `--allow-cpu` exists only for bounded code-path
tests and is not an accepted full-training mode.
