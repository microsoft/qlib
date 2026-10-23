# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader


@pytest.fixture(params=["pytorch_transformer", "pytorch_transformer_ts"])
def transformer_module(request):
    return importlib.import_module("qlib.contrib.model." + request.param)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_forward_keeps_batch_dimension(transformer_module, batch_size):
    model = transformer_module.Transformer(d_feat=2, d_model=4, nhead=2, num_layers=1, dropout=0)
    features = torch.randn(batch_size, 3, 2)
    if transformer_module.__name__.endswith("pytorch_transformer"):
        features = features.transpose(1, 2).reshape(batch_size, -1)

    predictions = model(features)

    assert predictions.shape == (batch_size,)
    loss = torch.nn.functional.mse_loss(predictions, torch.zeros(batch_size))
    loss.backward()
    assert model.feature_layer.weight.grad is not None
    assert torch.isfinite(model.feature_layer.weight.grad).all()
    assert model.decoder_layer.weight.grad is not None
    assert torch.isfinite(model.decoder_layer.weight.grad).all()


@pytest.mark.parametrize("sample_count", [1, 3, 4])
def test_predict_keeps_singleton_and_remainder_samples(transformer_module, sample_count):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=sample_count), ["SH600000"]], names=["datetime", "instrument"]
    )
    is_time_series = transformer_module.__name__.endswith("_ts")
    feature_count = 2 if is_time_series else 6
    columns = pd.MultiIndex.from_tuples([("feature", "f%d" % i) for i in range(feature_count)] + [("label", "LABEL0")])
    frame = pd.DataFrame(
        np.random.RandomState(0).normal(size=(sample_count, len(columns))).astype(np.float32),
        index=index,
        columns=columns,
    )
    handler = DataHandlerLP(data_loader=StaticDataLoader(frame))
    dataset_kwargs = {"handler": handler, "segments": {"test": ("2020-01-01", "2020-01-04")}}
    dataset = TSDatasetH(step_len=3, **dataset_kwargs) if is_time_series else DatasetH(**dataset_kwargs)
    model = transformer_module.TransformerModel(
        d_feat=2, d_model=4, nhead=2, num_layers=1, dropout=0, batch_size=2, n_jobs=0, GPU=-1, seed=0
    )
    # Exercise the real prediction wrapper without requiring a training run or market data.
    model.fitted = True

    predictions = model.predict(dataset)

    pd.testing.assert_index_equal(predictions.index, index)
    assert predictions.shape == (sample_count,)
    assert np.isfinite(predictions.to_numpy()).all()
    # Predictions must be independent of how samples are split into inference batches.
    model.batch_size = sample_count + 1
    unbatched = model.predict(dataset)
    np.testing.assert_allclose(predictions.to_numpy(), unbatched.to_numpy(), rtol=1e-5, atol=1e-6)
