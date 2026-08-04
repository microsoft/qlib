import torch

from qlib.contrib.model.pytorch_tcn_ts import TCNModel


def test_tcn_ts_model_keeps_batch_dimension_for_single_sample():
    model = TCNModel(num_input=3, output_size=1, num_channels=[4], kernel_size=2, dropout=0.0)

    assert model(torch.randn(1, 3, 8)).shape == (1,)
    assert model(torch.randn(2, 3, 8)).shape == (2,)
