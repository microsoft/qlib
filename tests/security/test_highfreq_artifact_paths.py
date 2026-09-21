from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qlib.contrib.data.highfreq_provider import HighFreqProvider
from qlib.utils.pickle_utils import ARTIFACT_MIGRATION_URL


def _provider(root):
    provider = object.__new__(HighFreqProvider)
    provider.artifact_root = Path(root).resolve()
    return provider


def test_highfreq_artifact_path_stays_under_root(tmp_path):
    provider = _provider(tmp_path)
    assert provider._resolve_artifact_path("data/features.pkl") == tmp_path / "data/features.pkl"


def test_highfreq_artifact_path_rejects_parent_traversal(tmp_path):
    provider = _provider(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="escapes artifact_root") as caught:
        provider._resolve_artifact_path("../outside.pkl")
    assert ARTIFACT_MIGRATION_URL in str(caught.value)
    assert "dedicated trusted artifact_root" in str(caught.value)


def test_highfreq_artifact_path_rejects_absolute_path(tmp_path):
    provider = _provider(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="escapes artifact_root"):
        provider._resolve_artifact_path(tmp_path / "outside.pkl")


def _symlink(link, target):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable")


@pytest.mark.parametrize(
    "method", ["_gen_data", "_gen_dataframe", "_gen_dataset", "_gen_day_dataset", "_gen_stock_dataset"]
)
def test_generation_rejects_escaping_path_without_mutating_config(tmp_path, method):
    provider = _provider(tmp_path / "artifacts")
    config = {"path": "../outside.pkl"}
    args = (config, "feature") if method in ("_gen_day_dataset", "_gen_stock_dataset") else (config,)
    with pytest.raises(ValueError, match="escapes artifact_root"):
        getattr(provider, method)(*args)
    assert config == {"path": "../outside.pkl"}
    assert not (tmp_path / "outside.pkl").exists()


@pytest.mark.parametrize("method", ["_gen_day_dataset", "_gen_stock_dataset"])
def test_temporary_dataset_rejects_symlink_escape(tmp_path, method):
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"must not be read")
    _symlink(root / "tmp_dataset.pkl", outside)
    with pytest.raises(ValueError, match="escapes artifact_root"):
        getattr(_provider(root), method)({"path": "."}, "feature")
    assert outside.read_bytes() == b"must not be read"


@pytest.mark.parametrize(
    "method, filename", [("_gen_dataframe", "featurestrain.pkl"), ("get_pre_datasets", "features_train.pkl")]
)
def test_split_dataset_rejects_symlink_escape(tmp_path, method, filename):
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"must not be overwritten")
    _symlink(root / filename, outside)
    provider = _provider(root)
    provider.feature_conf = {"path": "features.pkl"}
    provider.label_conf = {"path": "labels.pkl"}
    with pytest.raises(ValueError, match="escapes artifact_root"):
        if method == "get_pre_datasets":
            provider.get_pre_datasets()
        else:
            provider._gen_dataframe(provider.feature_conf)
    assert outside.read_bytes() == b"must not be overwritten"


def test_cached_data_load_preserves_input_config(tmp_path):
    expected = {"train": [1], "valid": [2], "test": [3]}
    (tmp_path / "features.pkl").write_bytes(pickle.dumps(expected))
    provider = _provider(tmp_path)
    provider.logger = Mock()
    config = {"path": "features.pkl"}
    assert provider._gen_data(config) == [[1], [2], [3]]
    assert config == {"path": "features.pkl"}


@pytest.mark.parametrize(
    "method, filename",
    [
        ("_gen_day_dataset", "2024-01-01.pkl"),
        ("_gen_stock_dataset", "SH600000.pkl"),
        ("_gen_stock_dataset", "../outside.pkl"),
    ],
)
def test_generated_dataset_rejects_escaping_filename(tmp_path, monkeypatch, method, filename):
    import pandas as pd
    from qlib.contrib.data import highfreq_provider as module

    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"must not be overwritten")
    if filename != "../outside.pkl":
        _symlink(root / filename, outside)
    (root / "tmp_dataset.pkl").write_bytes(pickle.dumps(None))
    provider = _provider(root)
    provider.logger = Mock()
    provider.start_time = provider.end_time = "2024-01-01"
    provider.freq = "1min"
    monkeypatch.setattr(
        module,
        "D",
        SimpleNamespace(
            calendar=lambda **kwargs: [pd.Timestamp("2024-01-01")],
            instruments=lambda **kwargs: [],
            list_instruments=lambda **kwargs: [filename[:-4]],
        ),
    )
    monkeypatch.setattr(module, "Parallel", lambda **kwargs: lambda jobs: [fun(*args, **kw) for fun, args, kw in jobs])
    with pytest.raises(ValueError, match="escapes artifact_root"):
        getattr(provider, method)({"path": "."}, "feature")
    assert outside.read_bytes() == b"must not be overwritten"
