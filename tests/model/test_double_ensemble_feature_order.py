# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from qlib.contrib.model.double_ensemble import DEnsembleModel


class SumBooster:
    def predict(self, values):
        return values.sum(axis=1)


def test_feature_selection_preserves_sampled_column_order(monkeypatch):
    """A fixed RNG result must not be reordered by Python's hash seed."""
    names = ["alpha", "beta", "gamma", "delta"]
    features = pd.DataFrame(np.arange(24).reshape(6, 4), columns=names)
    labels = pd.DataFrame({"label": features.sum(axis=1)})
    frame = pd.concat({"feature": features, "label": labels}, axis=1)
    model = DEnsembleModel(num_models=1, bins_fs=1, sample_ratios=[1.0])
    model.ensemble = [SumBooster()]
    model.sub_features = [pd.Index(names)]
    sampled = np.array(["gamma", "alpha", "delta", "beta"])

    def choose(values, size, replace):
        assert set(values) == set(names)
        assert size == len(names)
        assert replace is False
        return sampled.copy()

    monkeypatch.setattr(np.random, "choice", choose)
    selected = model.feature_selection(frame, pd.Series(np.zeros(len(frame))))

    assert selected.tolist() == sampled.tolist()
    assert selected.is_unique
