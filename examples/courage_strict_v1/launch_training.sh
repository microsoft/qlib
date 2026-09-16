#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

# Torch can load the host libstdc++ before MLflow imports sqlite.  Keep the
# project environment's ABI-compatible runtime first for every torchrun rank.
export LD_LIBRARY_PATH="${project_root}/.venv/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

exec "${project_root}/.venv/bin/torchrun" \
  --standalone \
  --nproc_per_node=8 \
  examples/courage_strict_v1/run_experiment.py \
  --batch-size 256 \
  --workers 8
