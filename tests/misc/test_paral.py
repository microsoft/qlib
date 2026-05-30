import qlib.utils.paral as paral_mod
from qlib.utils.paral import ParallelExt


def test_parallel_ext_uses_semantic_joblib_version(monkeypatch):
    monkeypatch.setattr(paral_mod.joblib, "__version__", "1.10.0")

    parallel = ParallelExt(n_jobs=2, backend="multiprocessing", maxtasksperchild=3)

    assert parallel._backend_kwargs["maxtasksperchild"] == 3
