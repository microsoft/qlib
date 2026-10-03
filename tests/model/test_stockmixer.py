# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class TestStockMixer(unittest.TestCase):
    """Unit tests for the StockMixer contrib model (synthetic data only)."""

    def setUp(self):
        try:
            import torch  # noqa: F401
            from qlib.contrib.model.pytorch_stockmixer import StockMixer, StockMixerModel  # noqa: F401
        except ImportError:
            print("Import error (torch not installed?).")
            raise unittest.SkipTest("pytorch is not installed")
        self.torch = torch
        self.StockMixer = StockMixer
        self.StockMixerModel = StockMixerModel

    @staticmethod
    def _benchmark_config(name):
        path = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "benchmarks"
            / "StockMixer"
            / ("workflow_config_stockmixer_%s.yaml" % name)
        )
        with open(path) as fp:
            return yaml.safe_load(fp)

    def test_instantiate_from_benchmark_configs(self):
        # the model must be constructible from both published workflow configs
        for name in ("Alpha360", "Alpha158"):
            cfg = self._benchmark_config(name)
            model_cfg = cfg["task"]["model"]
            self.assertEqual(model_cfg["class"], "StockMixer")
            self.assertEqual(model_cfg["module_path"], "qlib.contrib.model.pytorch_stockmixer")
            model = self.StockMixer(**model_cfg["kwargs"])
            self.assertFalse(model.fitted)

    def test_forward_backward(self):
        # both the Alpha360 layout (time_steps=60) and the Alpha158 layout (time_steps=1)
        torch = self.torch
        torch.manual_seed(0)
        for d_feat, time_steps in [(6, 60), (157, 1)]:
            model = self.StockMixerModel(stocks=10, channels=d_feat, time_steps=time_steps, market=20)
            output = model(torch.randn(10, d_feat * time_steps))
            self.assertEqual(tuple(output.shape), (10,))
            output.pow(2).mean().backward()

    def test_padding_mask_invariance(self):
        # changing the values of zero-padded rows must not change real rows' outputs
        torch = self.torch
        torch.manual_seed(0)
        for d_feat, time_steps in [(6, 60), (157, 1)]:
            model = self.StockMixerModel(stocks=10, channels=d_feat, time_steps=time_steps, market=20).eval()
            mask = torch.zeros(10, dtype=torch.bool)
            mask[:6] = True
            x = torch.randn(10, d_feat * time_steps)
            x_noisy = x.clone()
            x_noisy[6:] = torch.randn(4, d_feat * time_steps) * 100.0
            with torch.no_grad():
                out = model(x, mask)
                out_noisy = model(x_noisy, mask)
            self.assertTrue(torch.allclose(out[:6], out_noisy[:6]))
            # no-mask path (backward compatibility) must keep working
            self.assertEqual(tuple(model(x).shape), (10,))

    def test_n_stock_overflow_raises(self):
        model = self.StockMixer(d_feat=6, time_steps=60, n_stock=4, n_epochs=1, GPU=-1, seed=1)
        days = pd.date_range("2020-01-01", periods=2)
        inst = ["S%03d" % i for i in range(12)]
        index = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
        x = pd.DataFrame(np.random.randn(len(index), 360).astype(np.float32), index=index)
        y = pd.DataFrame(np.random.randn(len(index), 1).astype(np.float32), index=index)
        with self.assertRaises(ValueError):
            model.train_epoch(x, y)

    def test_fit_and_predict(self):
        # full wrapper flow on synthetic day-grouped cross-sections with padding
        np.random.seed(0)
        days = pd.date_range("2020-01-01", periods=8)
        inst = ["S%03d" % i for i in range(12)]
        index = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
        x = pd.DataFrame(np.random.randn(len(index), 360).astype(np.float32), index=index)
        y = pd.DataFrame(np.random.randn(len(index), 1).astype(np.float32), index=index, columns=["L0"])

        class StubDataset:
            def __init__(self, x, y):
                self._x = x
                self._df = pd.concat({"feature": x, "label": y}, axis=1)

            def prepare(self, segment, col_set="feature", data_key=None):
                if isinstance(segment, (list, tuple)):
                    return [self._df] * len(segment)
                return self._x

        ds = StubDataset(x, y)  # noqa: F841
        model = self.StockMixer(d_feat=6, time_steps=60, n_stock=16, n_epochs=2, lr=1e-3, GPU=-1, seed=42)
        model.fit(ds, save_path=None)
        pred = model.predict(ds, segment="test")
        self.assertEqual(pred.shape[0], len(index))
        self.assertTrue(pred.index.equals(index))
        self.assertTrue(np.isfinite(pred.values).all())


if __name__ == "__main__":
    unittest.main()
