# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import Text, Union
from qlib.log import get_module_logger
from qlib.data.dataset.weight import Reweighter
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class LinearModel(Model):
    """Linear Model

    Solve one of the following regression problems:
        - `ols`: min_w |y - Xw|^2_2
        - `nnls`: min_w |y - Xw|^2_2, s.t. w >= 0
        - `ridge`: min_w |y - Xw|^2_2 + \alpha*|w|^2_2
        - `lasso`: min_w |y - Xw|^2_2 + \alpha*|w|_1
    where `w` is the regression coefficient.
    """

    OLS = "ols"
    NNLS = "nnls"
    RIDGE = "ridge"
    LASSO = "lasso"

    def __init__(self, estimator="ols", alpha=0.0, fit_intercept=False, include_valid: bool = False):
        """
        Parameters
        ----------
        estimator : str
            which estimator to use for linear regression
        alpha : float
            l1 or l2 regularization parameter
        fit_intercept : bool
            whether fit intercept
        include_valid: bool
            Should the validation data be included for training?
            The validation data should be included
        """
        assert estimator in [self.OLS, self.NNLS, self.RIDGE, self.LASSO], f"unsupported estimator `{estimator}`"
        self.estimator = estimator

        assert alpha == 0 or estimator in [self.RIDGE, self.LASSO], f"alpha is only supported in `ridge`&`lasso`"
        self.alpha = alpha

        self.fit_intercept = fit_intercept

        self.coef_ = None
        self.include_valid = include_valid

    def fit(self, dataset: DatasetH, reweighter: Reweighter = None):
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if self.include_valid:
            try:
                df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                df_train = pd.concat([df_train, df_valid])
            except KeyError:
                get_module_logger("LinearModel").info("include_valid=True, but valid does not exist")
        if df_train.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        if reweighter is not None:
            w: pd.Series = reweighter.reweight(df_train)
            w = w.values
        else:
            w = None
        X, y = df_train["feature"].values, np.squeeze(df_train["label"].values)

        if self.estimator in [self.OLS, self.RIDGE, self.LASSO]:
            self._fit(X, y, w)
        elif self.estimator == self.NNLS:
            self._fit_nnls(X, y, w)
        else:
            raise ValueError(f"unknown estimator `{self.estimator}`")

        return self

    def _fit(self, X, y, w):
        if self.estimator == self.OLS:
            model = LinearRegression(fit_intercept=self.fit_intercept, copy_X=False)
        else:
            model = {self.RIDGE: Ridge, self.LASSO: Lasso}[self.estimator](
                alpha=self.alpha, fit_intercept=self.fit_intercept, copy_X=False
            )
        model.fit(X, y, sample_weight=w)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

    def _fit_nnls(self, X, y, w=None):
        if w is not None:
            raise NotImplementedError("TODO: support nnls with weight")  # TODO
        if self.fit_intercept:
            X = np.c_[X, np.ones(len(X))]  # NOTE: mem copy
        coef = nnls(X, y)[0]
        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = 0.0

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.coef_ is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(x_test.values @ self.coef_ + self.intercept_, index=x_test.index)
