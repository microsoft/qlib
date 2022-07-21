# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from email.mime import base
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from typing import Text, Union

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...data.dataset.weight import Reweighter


class StackingModel(Model, FeatureInt):
    """Stacking Model"""

    def __init__(self, **kwargs):
        self._params = {}
        self._params.update(kwargs)
        self.model = None

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        reweighter=None,
        **kwargs
    ):

        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Stacking base estimator need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("Stacking base estimator doesn't support multi-label training")

        n_estimators = self._params["n_estimators"]
        base_estimator = self._params["base_estimator"]
        if base_estimator == 'lightgbm':
            estimators = [
                (f'lightgbm_{i}', LGBMRegressor(
                        seed=random.randint(0, 1000),
                        **self._params["lightgbm"],
                        **kwargs
                    )
                )
                for i in range(n_estimators)
            ]
        else:
            raise NotImplementedError(f"Not supported base estimator {base_estimator}!")
            
        params = {}
        params["n_jobs"] = self._params["n_jobs"]
        params["verbose"] = self._params["verbose"]
        self.model = StackingRegressor(
            estimators=estimators,
            **params
        )
        self.model.fit(x_train.values, y_train_1d)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test), index=x_test.index)
