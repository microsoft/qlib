from pathlib import Path

import pytest

from qlib.contrib.data.highfreq_provider import HighFreqProvider


def _provider(root):
    provider = object.__new__(HighFreqProvider)
    provider.artifact_root = Path(root).resolve()
    return provider


def test_highfreq_artifact_path_stays_under_root(tmp_path):
    provider = _provider(tmp_path)
    assert provider._resolve_artifact_path("data/features.pkl") == tmp_path / "data/features.pkl"


def test_highfreq_artifact_path_rejects_parent_traversal(tmp_path):
    provider = _provider(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="escapes artifact_root"):
        provider._resolve_artifact_path("../outside.pkl")


def test_highfreq_artifact_path_rejects_absolute_path(tmp_path):
    provider = _provider(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="escapes artifact_root"):
        provider._resolve_artifact_path(tmp_path / "outside.pkl")
