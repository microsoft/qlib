# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error

from .base import Model
from ...utils import drop_nan_by_y_index


class LGBModel(Model):
    """LightGBM Model

    Parameters
    ----------
    param_update : dict
        training parameters
    """

    _params = dict()

    def __init__(self, loss="mse", **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self._scorer = mean_squared_error if loss == "mse" else roc_auc_score
        self._params.update(objective=loss, **kwargs)
        self._model = None

    def fit(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        w_train=None,
        w_valid=None,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        **kwargs
    ):
        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        w_train_weight = None if w_train is None else w_train.values
        w_valid_weight = None if w_valid is None else w_valid.values

        dtrain = lgb.Dataset(x_train.values, label=y_train_1d, weight=w_train_weight)
        dvalid = lgb.Dataset(x_valid.values, label=y_valid_1d, weight=w_valid_weight)
        self._model = lgb.train(
            self._params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            **kwargs
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

    def predict(self, x_test):
        if self._model is None:
            raise ValueError("model is not fitted yet!")
        return self._model.predict(x_test.values)

    def score(self, x_test, y_test, w_test=None):
        # Remove rows from x, y and w, which contain Nan in any columns in y_test.
        x_test, y_test, w_test = drop_nan_by_y_index(x_test, y_test, w_test)
        preds = self.predict(x_test)
        w_test_weight = None if w_test is None else w_test.values
        return self._scorer(y_test.values, preds, sample_weight=w_test_weight)

    def save(self, filename):
        if self._model is None:
            raise ValueError("model is not fitted yet!")
        self._model.save_model(filename)

    def load(self, buffer):
        self._model = lgb.Booster(params={"model_str": buffer.decode("utf-8")})
