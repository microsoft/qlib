# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

try:
    import qlib.contrib.model.catboost_model as catboost_model
except ImportError:
    catboost_model = None

try:
    import qlib.contrib.model.xgboost as xgboost_model
except ImportError:
    xgboost_model = None


class _Column:
    def __init__(self, values):
        self.values = np.asarray(values)


class _Frame:
    empty = False

    def __init__(self):
        self._columns = {
            "feature": _Column([[1.0], [2.0]]),
            "label": _Column([[0.0], [1.0]]),
        }

    def __getitem__(self, key):
        return self._columns[key]


class _Dataset:
    def prepare(self, *args, **kwargs):
        return _Frame(), _Frame()


@unittest.skipUnless(catboost_model is not None, "CatBoost dependencies are not installed")
class TestCatBoostModelEvalsResult(unittest.TestCase):
    def test_fit_updates_caller_evals_result(self):
        class FakeCatBoost:
            def fit(self, *args, **kwargs):
                pass

            def get_evals_result(self):
                return {
                    "learn": {"RMSE": [0.4, 0.2]},
                    "validation": {"RMSE": [0.5, 0.3]},
                }

        evals_result = {"existing": [1]}
        model = catboost_model.CatBoostModel()
        with mock.patch.object(catboost_model, "Pool"), mock.patch.object(
            catboost_model, "CatBoost", return_value=FakeCatBoost()
        ), mock.patch.object(catboost_model, "get_gpu_device_count", return_value=0):
            model.fit(_Dataset(), evals_result=evals_result)

        self.assertEqual(evals_result["train"], [0.4, 0.2])
        self.assertEqual(evals_result["valid"], [0.5, 0.3])
        self.assertEqual(evals_result["existing"], [1])

    def test_evals_result_default_is_not_mutable(self):
        default = inspect.signature(catboost_model.CatBoostModel.fit).parameters["evals_result"].default
        self.assertIsNone(default)


@unittest.skipUnless(xgboost_model is not None, "XGBoost dependencies are not installed")
class TestXGBModelEvalsResult(unittest.TestCase):
    def test_fit_updates_caller_evals_result(self):
        def train(*args, evals_result, **kwargs):
            evals_result.update({"train": {"rmse": [0.4]}, "valid": {"rmse": [0.5]}})
            return object()

        evals_result = {"existing": [1]}
        model = xgboost_model.XGBModel()
        with mock.patch.object(xgboost_model.xgb, "DMatrix", return_value=object()), mock.patch.object(
            xgboost_model.xgb, "train", side_effect=train
        ):
            model.fit(_Dataset(), evals_result=evals_result)

        self.assertEqual(evals_result["train"], [0.4])
        self.assertEqual(evals_result["valid"], [0.5])
        self.assertEqual(evals_result["existing"], [1])

    def test_fit_uses_a_fresh_default_evals_result(self):
        captured = []

        def train(*args, evals_result, **kwargs):
            captured.append(evals_result)
            evals_result.update({"train": {"rmse": [0.4]}, "valid": {"rmse": [0.5]}})
            return object()

        model = xgboost_model.XGBModel()
        with mock.patch.object(xgboost_model.xgb, "DMatrix", return_value=object()), mock.patch.object(
            xgboost_model.xgb, "train", side_effect=train
        ):
            model.fit(_Dataset())
            model.fit(_Dataset())

        self.assertIsNot(captured[0], captured[1])

    def test_evals_result_default_is_not_mutable(self):
        default = inspect.signature(xgboost_model.XGBModel.fit).parameters["evals_result"].default
        self.assertIsNone(default)


if __name__ == "__main__":
    unittest.main()
