import json

import numpy as np
import pytest

pytest.importorskip("torch")

from qlib.contrib.model.pytorch_hist import _load_stock_index


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
