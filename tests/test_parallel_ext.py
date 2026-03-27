"""Test for Issue #1927: ParallelExt _backend_kwargs attribute fix."""
import pytest
from joblib import delayed
from qlib.utils.paral import ParallelExt


def test_parallel_ext_with_maxtasksperchild():
    """ParallelExt should accept maxtasksperchild without AttributeError."""
    p = ParallelExt(n_jobs=1, backend="loky", maxtasksperchild=10)
    results = p(delayed(lambda x: x * 2)(i) for i in range(5))
    assert results == [0, 2, 4, 6, 8]


def test_parallel_ext_without_maxtasksperchild():
    """ParallelExt should work normally without maxtasksperchild."""
    p = ParallelExt(n_jobs=1)
    results = p(delayed(sum)([i, 1]) for i in range(3))
    assert results == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
