# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from ...model.base import ModelFT
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import LightGBMFInt


class HFLGBModel(ModelFT, LightGBMFInt):
    """LightGBM Model for high frequency prediction"""

    def __init__(self, loss="mse", **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.model = None

    def _cal_signal_metrics(self, y_test, l_cut, r_cut):
        """
        Calcaute the signal metrics by daily level
        """
        up_pre, down_pre = [], []
        up_alpha_ll, down_alpha_ll = [], []
        for date in y_test.index.get_level_values(0).unique():
            df_res = y_test.loc[date].sort_values("pred")
            if int(l_cut * len(df_res)) < 10:
                warnings.warn("Warning: threhold is too low or instruments number is not enough")
                continue
            top = df_res.iloc[: int(l_cut * len(df_res))]
            bottom = df_res.iloc[int(r_cut * len(df_res)) :]

            down_precision = len(top[top[top.columns[0]] < 0]) / (len(top))
            up_precision = len(bottom[bottom[top.columns[0]] > 0]) / (len(bottom))

            down_alpha = top[top.columns[0]].mean()
            up_alpha = bottom[bottom.columns[0]].mean()

            up_pre.append(up_precision)
            down_pre.append(down_precision)
            up_alpha_ll.append(up_alpha)
            down_alpha_ll.append(down_alpha)

        return (
            np.array(up_pre).mean(),
            np.array(down_pre).mean(),
            np.array(up_alpha_ll).mean(),
            np.array(down_alpha_ll).mean(),
        )

    def hf_signal_test(self, dataset: DatasetH, threhold=0.2):
        """
        Test the signal in high frequency test set
        """
        if self.model == None:
            raise ValueError("Model hasn't been trained yet")
        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        df_test.dropna(inplace=True)
        x_test, y_test = df_test["feature"], df_test["label"]
        # Convert label into alpha
        y_test[y_test.columns[0]] = y_test[y_test.columns[0]] - y_test[y_test.columns[0]].mean(level=0)

        res = pd.Series(self.model.predict(x_test.values), index=x_test.index)
        y_test["pred"] = res

        up_p, down_p, up_a, down_a = self._cal_signal_metrics(y_test, threhold, 1 - threhold)
        print("===============================")
        print("High frequency signal test")
        print("===============================")
        print("Test set precision: ")
        print("Positive precision: {}, Negative precision: {}".format(up_p, down_p))
        print("Test Alpha Average in test set: ")
        print("Positive average alpha: {}, Negative average alpha: {}".format(up_a, down_a))

    def _prepare_data(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            l_name = df_train["label"].columns[0]
            # Convert label into alpha
            df_train["label"][l_name] = df_train["label"][l_name] - df_train["label"][l_name].mean(level=0)
            df_valid["label"][l_name] = df_valid["label"][l_name] - df_valid["label"][l_name].mean(level=0)
            mapping_fn = lambda x: 0 if x < 0 else 1
            df_train["label_c"] = df_train["label"][l_name].apply(mapping_fn)
            df_valid["label_c"] = df_valid["label"][l_name].apply(mapping_fn)
            x_train, y_train = df_train["feature"], df_train["label_c"].values
            x_valid, y_valid = df_valid["feature"], df_valid["label_c"].values
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        return dtrain, dvalid

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        **kwargs
    ):
        dtrain, dvalid = self._prepare_data(dataset)
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

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20):
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
        dtrain, _ = self._prepare_data(dataset)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            verbose_eval=verbose_eval,
        )
