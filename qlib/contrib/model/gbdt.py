# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Text, Union
from ...model.base import ModelFT
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import LightGBMFInt
from ...data.dataset.weight import Reweighter


class LGBModel(ModelFT, LightGBMFInt):
    """LightGBM Model"""

    def __init__(self, loss="mse", **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.model = None

    def _prepare_data(self, dataset: DatasetH, reweighter=None):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            w_train = reweighter.reweight(df_train)
            w_valid = reweighter.reweight(df_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        dtrain = lgb.Dataset(x_train.values, label=y_train, weight=w_train)
        dvalid = lgb.Dataset(x_valid.values, label=y_valid, weight=w_valid)
        return dtrain, dvalid

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        reweighter=None,
        **kwargs
    ):
        dtrain, dvalid = self._prepare_data(dataset, reweighter)
        self.model = lgb.train(
            self.params,
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

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20, reweighter=None):
        """
        finetune model

        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        num_boost_round : int
            number of round to finetune model
        verbose_eval : int
            verbose level
        """
        # Based on existing model and finetune by train more rounds
        dtrain, _ = self._prepare_data(dataset, reweighter)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            verbose_eval=verbose_eval,
        )
