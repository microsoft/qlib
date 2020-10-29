# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import lightgbm as lgb

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class LGBModel(Model):
    """LightGBM Model"""
    def __init__(self, loss="mse", **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self._params = {'objective': loss}
        self._params.update(kwargs)
        self.model = None

    def fit(self,
            dataset: DatasetH,
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=20,
            evals_result=dict(),
            **kwargs):

        df_train, df_valid = dataset.prepare(['train', 'valid'],
                                             col_set=['feature', 'label'],
                                             data_key=DataHandlerLP.DK_L)
        x_train, y_train = df_train['feature'], df_train['label']
        x_valid, y_valid = df_valid['feature'], df_valid['label']

        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        dtrain = lgb.Dataset(x_train.values, label=y_train_1d)
        dvalid = lgb.Dataset(x_valid.values, label=y_valid_1d)
        self.model = lgb.train(self._params,
                               dtrain,
                               num_boost_round=num_boost_round,
                               valid_sets=[dtrain, dvalid],
                               valid_names=["train", "valid"],
                               early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=verbose_eval,
                               evals_result=evals_result,
                               **kwargs)
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare('test', col_set='feature')
        return pd.Series(self.model.predict(np.squeeze(x_test.values)), index=x_test.index)
