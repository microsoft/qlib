# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Text, Union
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...data.dataset.weight import Reweighter
from ...log import get_module_logger

from tensorboardX import SummaryWriter


class XGBModel(Model, FeatureInt):
    """XGBModel Model"""

    def __init__(self, tensorboard=False, tensorboard_name="", **kwargs):
        self.logger = get_module_logger("XGBoostModel")
        self._params = {}
        self._params.update(kwargs)
        self.model = None
        self.tensorboard = tensorboard
        self.tensorboard_name = tensorboard_name

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
        **kwargs,
    ):

        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("XGBoost doesn't support multi-label training")

        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            w_train = reweighter.reweight(df_train)
            w_valid = reweighter.reweight(df_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        dtrain = xgb.DMatrix(x_train.values, label=y_train_1d, weight=w_train)
        dvalid = xgb.DMatrix(x_valid.values, label=y_valid_1d, weight=w_valid)

        # Activate tensorboard
        callbacks = []
        if self.tensorboard:
            callbacks = [TensorBoardCallback(experiment=self.tensorboard_name, data_name="valid")]

        self.model = xgb.train(
            self._params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            callbacks=callbacks,
            **kwargs,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(xgb.DMatrix(x_test)), index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)


class TensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, experiment: str = None, data_name: str = None):
        working_dir = os.getcwd()
        target_dir = os.path.join(os.path.relpath(working_dir), "tensor_board/")
        sys.path.insert(0, target_dir)

        self.experiment = experiment or "logs"
        self.data_name = data_name or "test"
        self.log_dir = f"runs/xgboost/{self.experiment}"
        self.train_writer = SummaryWriter(log_dir=target_dir + os.path.join(self.log_dir, "train/"))
        if self.data_name:
            self.test_writer = SummaryWriter(log_dir=target_dir + os.path.join(self.log_dir, f"{self.data_name}/"))

    def after_iteration(self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog) -> bool:
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    self.train_writer.add_scalar(metric_name, score, epoch)
                else:
                    self.test_writer.add_scalar(metric_name, score, epoch)

        return False
