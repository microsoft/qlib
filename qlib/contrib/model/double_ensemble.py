# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Text, Union
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import FeatureInt
from ...log import get_module_logger


class DEnsembleModel(Model, FeatureInt):
    """Double Ensemble Model"""

    def __init__(
        self,
        base_model="gbm",
        loss="mse",
        num_models=6,
        enable_sr=True,
        enable_fs=True,
        alpha1=1.0,
        alpha2=1.0,
        bins_sr=10,
        bins_fs=5,
        decay=None,
        sample_ratios=None,
        sub_weights=None,
        epochs=100,
        early_stopping_rounds=None,
        **kwargs
    ):
        self.base_model = base_model  # "gbm" or "mlp", specifically, we use lgbm for "gbm"
        self.num_models = num_models  # the number of sub-models
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.decay = decay
        if sample_ratios is None:  # the default values for sample_ratios
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:  # the default values for sub_weights
            sub_weights = [1] * self.num_models
        if not len(sample_ratios) == bins_fs:
            raise ValueError("The length of sample_ratios should be equal to bins_fs.")
        self.sample_ratios = sample_ratios
        if not len(sub_weights) == num_models:
            raise ValueError("The length of sub_weights should be equal to num_models.")
        self.sub_weights = sub_weights
        self.epochs = epochs
        self.logger = get_module_logger("DEnsembleModel")
        self.logger.info("Double Ensemble Model...")
        self.ensemble = []  # the current ensemble model, a list contains all the sub-models
        self.sub_features = []  # the features for each sub model in the form of pandas.Index
        self.params = {"objective": loss}
        self.params.update(kwargs)
        self.loss = loss
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        x_train, y_train = df_train["feature"], df_train["label"]
        # initialize the sample weights
        N, F = x_train.shape
        weights = pd.Series(np.ones(N, dtype=float))
        # initialize the features
        features = x_train.columns
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models), dtype=float), index=x_train.index)
        # train sub-models
        for k in range(self.num_models):
            self.sub_features.append(features)
            self.logger.info("Training sub-model: ({}/{})".format(k + 1, self.num_models))
            model_k = self.train_submodel(df_train, df_valid, weights, features)
            self.ensemble.append(model_k)
            # no further sample re-weight and feature selection needed for the last sub-model
            if k + 1 == self.num_models:
                break

            self.logger.info("Retrieving loss curve and loss values...")
            loss_curve = self.retrieve_loss_curve(model_k, df_train, features)
            pred_k = self.predict_sub(model_k, df_train, features)
            pred_sub.iloc[:, k] = pred_k
            pred_ensemble = (pred_sub.iloc[:, : k + 1] * self.sub_weights[0 : k + 1]).sum(axis=1) / np.sum(
                self.sub_weights[0 : k + 1]
            )
            loss_values = pd.Series(self.get_loss(y_train.values.squeeze(), pred_ensemble.values))

            if self.enable_sr:
                self.logger.info("Sample re-weighting...")
                weights = self.sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                self.logger.info("Feature selection...")
                features = self.feature_selection(df_train, loss_values)

    def train_submodel(self, df_train, df_valid, weights, features):
        dtrain, dvalid = self._prepare_data_gbm(df_train, df_valid, weights, features)
        evals_result = dict()

        callbacks = [lgb.log_evaluation(20), lgb.record_evaluation(evals_result)]
        if self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            self.logger.info("Training with early_stopping...")

        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.epochs,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]
        return model

    def _prepare_data_gbm(self, df_train, df_valid, weights, features):
        x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
        x_valid, y_valid = df_valid["feature"].loc[:, features], df_valid["label"]

        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        dtrain = lgb.Dataset(x_train, label=y_train, weight=weights)
        dvalid = lgb.Dataset(x_valid, label=y_valid)
        return dtrain, dvalid

    def sample_reweight(self, loss_curve, loss_values, k_th):
        """
        the SR module of Double Ensemble
        :param loss_curve: the shape is NxT
        the loss curve for the previous sub-model, where the element (i, t) if the error on the i-th sample
        after the t-th iteration in the training of the previous sub-model.
        :param loss_values: the shape is N
        the loss of the current ensemble on the i-th sample.
        :param k_th: the index of the current sub-model, starting from 1
        :return: weights
        the weights for all the samples.
        """
        # normalize loss_curve and loss_values with ranking
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        loss_values_norm = (-loss_values).rank(pct=True)

        # calculate l_start and l_end from loss_curve
        N, T = loss_curve.shape
        part = np.maximum(int(T * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        # calculate h-value for each sample
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})

        # calculate weights
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins")["h_value"].mean()
        weights = pd.Series(np.zeros(N, dtype=float))
        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.decay**k_th * h_avg[b] + 0.1)
        return weights

    def feature_selection(self, df_train, loss_values):
        """
        the FS module of Double Ensemble
        :param df_train: the shape is NxF
        :param loss_values: the shape is N
        the loss of the current ensemble on the i-th sample.
        :return: res_feat: in the form of pandas.Index

        """
        x_train, y_train = df_train["feature"], df_train["label"]
        features = x_train.columns
        N, F = x_train.shape
        g = pd.DataFrame({"g_value": np.zeros(F, dtype=float)})
        M = len(self.ensemble)

        # shuffle specific columns and calculate g-value for each feature
        x_train_tmp = x_train.copy()
        for i_f, feat in enumerate(features):
            x_train_tmp.loc[:, feat] = np.random.permutation(x_train_tmp.loc[:, feat].values)
            pred = pd.Series(np.zeros(N), index=x_train_tmp.index)
            for i_s, submodel in enumerate(self.ensemble):
                pred += (
                    pd.Series(
                        submodel.predict(x_train_tmp.loc[:, self.sub_features[i_s]].values), index=x_train_tmp.index
                    )
                    / M
                )
            loss_feat = self.get_loss(y_train.values.squeeze(), pred.values)
            g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)
            x_train_tmp.loc[:, feat] = x_train.loc[:, feat].copy()

        # one column in train features is all-nan # if g['g_value'].isna().any()
        g["g_value"].replace(np.nan, 0, inplace=True)

        # divide features into bins_fs bins
        g["bins"] = pd.cut(g["g_value"], self.bins_fs)

        # randomly sample features from bins to construct the new features
        res_feat = []
        sorted_bins = sorted(g["bins"].unique(), reverse=True)
        for i_b, b in enumerate(sorted_bins):
            b_feat = features[g["bins"] == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()
        return pd.Index(set(res_feat))

    def get_loss(self, label, pred):
        if self.loss == "mse":
            return (label - pred) ** 2
        else:
            raise ValueError("not implemented yet")

    def retrieve_loss_curve(self, model, df_train, features):
        if self.base_model == "gbm":
            num_trees = model.num_trees()
            x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
            # Lightgbm need 1D array as its label
            if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
                y_train = np.squeeze(y_train.values)
            else:
                raise ValueError("LightGBM doesn't support multi-label training")

            N = x_train.shape[0]
            loss_curve = pd.DataFrame(np.zeros((N, num_trees)))
            pred_tree = np.zeros(N, dtype=float)
            for i_tree in range(num_trees):
                pred_tree += model.predict(x_train.values, start_iteration=i_tree, num_iteration=1)
                loss_curve.iloc[:, i_tree] = self.get_loss(y_train, pred_tree)
        else:
            raise ValueError("not implemented yet")
        return loss_curve

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.ensemble is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        pred = pd.Series(np.zeros(x_test.shape[0]), index=x_test.index)
        for i_sub, submodel in enumerate(self.ensemble):
            feat_sub = self.sub_features[i_sub]
            pred += (
                pd.Series(submodel.predict(x_test.loc[:, feat_sub].values), index=x_test.index)
                * self.sub_weights[i_sub]
            )
        pred = pred / np.sum(self.sub_weights)
        return pred

    def predict_sub(self, submodel, df_data, features):
        x_data = df_data["feature"].loc[:, features]
        pred_sub = pd.Series(submodel.predict(x_data.values), index=x_data.index)
        return pred_sub

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -----
            parameters reference:
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance
        """
        res = []
        for _model, _weight in zip(self.ensemble, self.sub_weights):
            res.append(pd.Series(_model.feature_importance(*args, **kwargs), index=_model.feature_name()) * _weight)
        return pd.concat(res, axis=1, sort=False).sum(axis=1).sort_values(ascending=False)
