# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys

import numpy as np
import pandas as pd
from typing import Text, Union
from catboost import Pool, CatBoost
from catboost.utils import get_gpu_device_count

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...data.dataset.weight import Reweighter
from ...log import get_module_logger

working_dir = os.getcwd()
target_dir = os.path.join(os.path.relpath(working_dir), "tensor_board")
target_dir += "/runs/catboost/"
sys.path.insert(0, target_dir)


class CatBoostModel(Model, FeatureInt):
    """CatBoost Model"""

    def __init__(self, loss="RMSE", tensorboard=False, tensorboard_name="", **kwargs):
        self.logger = get_module_logger("CatBoostModel")

        # There are more options
        if loss not in {"RMSE", "Logloss"}:
            raise NotImplementedError
        self._params = {"loss_function": loss}
        self._params.update(kwargs)
        self.model = None
        self.tensorboard = tensorboard
        self.tensorboard_name = tensorboard_name
        # Log info
        self.logger.info("model parameters:\n{:}".format(self._params))
        self.logger.info("Tensorboard: {}".format(self.tensorboard))
        self.logger.info("Tensorboard Name: {}".format(self.tensorboard_name))

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
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # CatBoost needs 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("CatBoost doesn't support multi-label training")

        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            w_train = reweighter.reweight(df_train).values
            w_valid = reweighter.reweight(df_valid).values
        else:
            raise ValueError("Unsupported reweighter type.")

        train_pool = Pool(data=x_train, label=y_train_1d, weight=w_train)
        valid_pool = Pool(data=x_valid, label=y_valid_1d, weight=w_valid)

        # Initialize the catboost model
        self._params["iterations"] = num_boost_round
        self._params["early_stopping_rounds"] = early_stopping_rounds
        self._params["verbose_eval"] = verbose_eval
        self._params["task_type"] = "GPU" if get_gpu_device_count() > 0 else "CPU"
        if self.tensorboard:
            # The directory for storing the files generated during training.
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            self._params["train_dir"] = target_dir + self.tensorboard_name
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
