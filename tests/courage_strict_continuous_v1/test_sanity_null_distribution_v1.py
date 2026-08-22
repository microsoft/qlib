from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = (
    ROOT
    / "examples/courage_strict_continuous_v1/run_sanity_null_distribution_v1.py"
)
SPEC = importlib.util.spec_from_file_location("sanity_null_v1", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_circular_block_sample_preserves_length_and_values() -> None:
    values = np.arange(19, dtype=np.float64)
    result = MODULE._circular_block_sample(
        values, block_length=3, rng=np.random.default_rng(9)
    )
    assert len(result) == len(values)
    assert set(result).issubset(set(values))


def test_bootstrap_ci_is_reproducible_and_centred() -> None:
    values = {
        seed: np.full(19, seed / 100.0, dtype=np.float64) for seed in range(10)
    }
    first = MODULE._bootstrap_ci(
        values,
        block_length=3,
        replications=500,
        rng=np.random.default_rng(11),
    )
    second = MODULE._bootstrap_ci(
        values,
        block_length=3,
        replications=500,
        rng=np.random.default_rng(11),
    )
    assert first == second
    assert np.isclose(first[0], np.mean(np.arange(10) / 100.0))
    assert first[1] < first[0] < first[2]
