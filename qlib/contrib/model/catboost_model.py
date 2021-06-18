# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import Text, Union
from catboost import Pool, CatBoost
from catboost.utils import get_gpu_device_count

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt


class CatBoostModel(Model, FeatureInt):
    """CatBoost Model"""

    def __init__(self, loss="RMSE", **kwargs):
        # There are more options
        if loss not in {"RMSE", "Logloss"}:
            raise NotImplementedError
        self._params = {"loss_function": loss}
        self._params.update(kwargs)
        self.model = None

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        **kwargs
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # CatBoost needs 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("CatBoost doesn't support multi-label training")

        train_pool = Pool(data=x_train, label=y_train_1d)
        valid_pool = Pool(data=x_valid, label=y_valid_1d)

        # Initialize the catboost model
        self._params["iterations"] = num_boost_round
        self._params["early_stopping_rounds"] = early_stopping_rounds
        self._params["verbose_eval"] = verbose_eval
        self._params["task_type"] = "GPU" if get_gpu_device_count() > 0 else "CPU"
        self.model = CatBoost(self._params, **kwargs)

        # train the model
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=True, **kwargs)

        evals_result = self.model.get_evals_result()
        evals_result["train"] = list(evals_result["learn"].values())[0]
        evals_result["valid"] = list(evals_result["validation"].values())[0]

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -----
            parameters references:
            https://catboost.ai/docs/concepts/python-reference_catboost_get_feature_importance.html#python-reference_catboost_get_feature_importance
        """
        return pd.Series(
            data=self.model.get_feature_importance(*args, **kwargs), index=self.model.feature_names_
        ).sort_values(ascending=False)


if __name__ == "__main__":
    cat = CatBoostModel()
