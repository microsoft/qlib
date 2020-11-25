# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

class TabNetModel(Model):
    """TabNetModel Model"""

    def __init__(self, n_d, n_a, 
                        n_steps, 
                        gamma, 
                        n_independent, 
                        n_shared, 
                        seed, 
                        momentum, 
                        lambda_sparse, 
                        optimizer_params, 
                        **kwargs):
        self.model = None

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.seed = seed
        self.momentum = momentum
        self.lambda_sparse = lambda_sparse
        self.optimizer_params = optimizer_params

    def fit(
        self,
        dataset: DatasetH,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        seed=0,
        momentum=0.02,
        lambda_sparse=1e-3,
        optimizer_params={'lr':2e-3},
        **kwargs
    ):

        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        x_train, y_train = df_train["feature"].values, df_train["label"].values*100
        x_valid, y_valid = df_valid["feature"].values, df_valid["label"].values*100

        self.model = TabNetRegressor(
                    n_d=self.n_d,
                    n_a=self.n_a,
                    n_steps=self.n_steps,
                    gamma=self.gamma,
                    n_independent=self.n_independent,
                    n_shared=self.n_shared,
                    seed=self.seed,
                    momentum=self.momentum,
                    lambda_sparse=self.lambda_sparse,
                    optimizer_params=self.optimizer_params,
                    **kwargs
        )
        self.model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare("test", col_set="feature")
        test_pred = self.model.predict(x_test.values)
        return pd.Series(test_pred.reshape([-1]), index=x_test.index)
