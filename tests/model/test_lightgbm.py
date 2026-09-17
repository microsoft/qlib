# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for LightGBM API callback usage and model compatibility."""

import os
import sys
import types
import unittest

# Ensure mlflow file store is permitted in newer mlflow versions
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Mock Cython modules if they are not compiled
for mod_name, funcs in [
    ("qlib.data._libs.rolling", ["rolling_slope", "rolling_rsquare", "rolling_resi"]),
    ("qlib.data._libs.expanding", ["expanding_slope", "expanding_rsquare", "expanding_resi"]),
]:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            mod = types.ModuleType(mod_name)
            for f in funcs:
                setattr(mod, f, lambda *a, **k: None)
            sys.modules[mod_name] = mod

import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position
import qlib  # pylint: disable=wrong-import-position
from qlib.contrib.model.gbdt import LGBModel  # pylint: disable=wrong-import-position
from qlib.contrib.model.highfreq_gdbt_model import HFLGBModel  # pylint: disable=wrong-import-position
from qlib.workflow import R  # pylint: disable=wrong-import-position


class MockDataset:  # pylint: disable=too-few-public-methods
    """Mock dataset class for LightGBM model unit tests."""

    segments = ["train", "valid", "test"]

    def _get_df(self, segment, col_set=None):
        n = 100 if segment == "train" else 30
        dates = pd.date_range("2020-01-01", periods=n)
        index = pd.MultiIndex.from_arrays(
            [dates, ["SH600000"] * n],
            names=["datetime", "instrument"],
        )
        cols = pd.MultiIndex.from_tuples(
            [("feature", "f1"), ("feature", "f2"), ("label", "target")],
        )
        data = np.random.randn(n, 3)
        df = pd.DataFrame(data, index=index, columns=cols)
        if col_set == "feature":
            return df["feature"]
        return df

    def prepare(self, segments, col_set=None, data_key=None):  # pylint: disable=unused-argument
        """Prepare mock dataset splits."""
        if isinstance(segments, (list, tuple)):
            return [self._get_df(s, col_set) for s in segments]
        return self._get_df(segments, col_set)


class TestLightGBM(unittest.TestCase):
    """Test suite for LightGBM models with modern callback API."""

    @classmethod
    def setUpClass(cls):
        qlib.init()

    def test_lgb_model(self):
        """Test LGBModel fit, predict, and finetune."""
        with R.start(experiment_name="test_lgb"):
            dataset = MockDataset()
            model = LGBModel(early_stopping_rounds=10, num_boost_round=20)
            model.fit(dataset)
            pred = model.predict(dataset)
            self.assertEqual(len(pred), 30)
            model.finetune(dataset, num_boost_round=5)

    def test_lgb_model_no_early_stopping(self):
        """Test LGBModel when early_stopping_rounds is None."""
        with R.start(experiment_name="test_lgb_no_es"):
            dataset = MockDataset()
            model = LGBModel(early_stopping_rounds=None, num_boost_round=10)
            model.fit(dataset)
            pred = model.predict(dataset)
            self.assertEqual(len(pred), 30)

    def test_lgb_model_custom_callbacks(self):
        """Test LGBModel with custom callbacks passed via kwargs."""
        with R.start(experiment_name="test_lgb_cb"):
            dataset = MockDataset()
            called = []

            def custom_cb(env):
                called.append(env.iteration)

            model = LGBModel(early_stopping_rounds=10, num_boost_round=10)
            model.fit(dataset, callbacks=[custom_cb])
            self.assertGreater(len(called), 0)

    def test_hflgb_model(self):
        """Test HFLGBModel fit, predict, and finetune."""
        dataset = MockDataset()
        hf_model = HFLGBModel(early_stopping_rounds=10, num_boost_round=20)
        hf_model.fit(dataset)
        hf_pred = hf_model.predict(dataset)
        self.assertEqual(len(hf_pred), 30)
        hf_model.finetune(dataset, num_boost_round=5)


if __name__ == "__main__":
    unittest.main()
