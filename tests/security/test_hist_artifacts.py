import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from qlib.contrib.model.pytorch_hist import HIST, _load_stock_index


def test_load_stock_index_from_json(tmp_path):
    path = tmp_path / "stock_index.json"
    path.write_text(json.dumps({"SH600000": 0, "SZ000001": 1}), encoding="utf-8")

    assert _load_stock_index(path, upper_bound=2) == {"SH600000": 0, "SZ000001": 1}


@pytest.mark.parametrize("value", [-1, 2, 1.5, True, None])
def test_load_stock_index_rejects_invalid_values(tmp_path, value):
    path = tmp_path / "stock_index.json"
    path.write_text(json.dumps({"SH600000": value}), encoding="utf-8")

    with pytest.raises(ValueError):
        _load_stock_index(path, upper_bound=2)


def test_load_stock_index_rejects_object_npy(tmp_path):
    path = tmp_path / "stock_index.npy"
    np.save(path, {"SH600000": 0}, allow_pickle=True)

    with pytest.raises(ValueError, match="must be a JSON file"):
        _load_stock_index(path)


class _MarkerPayload:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return eval, (f"open({self.path!r}, 'w').write('executed')",)


@pytest.mark.parametrize("method", ["fit", "predict"])
def test_hist_rejects_malicious_metadata_before_execution(tmp_path, method):
    marker = tmp_path / "executed.txt"
    metadata = tmp_path / "index.npy"
    np.save(metadata, {"SH600000": _MarkerPayload(marker)}, allow_pickle=True)
    concepts = tmp_path / "concepts.npy"
    np.save(concepts, np.zeros((734, 1)), allow_pickle=False)
    model = object.__new__(HIST)
    model.stock_index = metadata
    model.stock2concept = concepts
    model.fitted = True
    dataset = Mock()
    dataset.prepare.return_value = [pd.DataFrame({"value": [1]})] * 3
    with pytest.raises(ValueError, match="must be a JSON file"):
        getattr(model, method)(dataset)
    assert not marker.exists()
