# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import Union
from sklearn.decomposition import PCA, FactorAnalysis

from qlib.model.riskmodel import RiskModel


class StructuredCovEstimator(RiskModel):
    """Structured Covariance Estimator

    This estimator assumes observations can be predicted by multiple factors
        X = B @ F.T + U
    where `X` contains observations (row) of multiple variables (column),
    `F` contains factor exposures (column) for all variables (row),
    `B` is the regression coefficients matrix for all observations (row) on
    all factors (columns), and `U` is the residual matrix with shape like `X`.

    Therefore the structured covariance can be estimated by
        cov(X.T) = F @ cov(B.T) @ F.T + diag(var(U))

    In finance domain, there are mainly three methods to design `F` [1][2]:
        - Statistical Risk Model (SRM): latent factor models major components
        - Fundamental Risk Model (FRM): human designed factors
        - Deep Risk Model (DRM): neural network designed factors (like a blend of SRM & DRM)

    In this implementation we use latent factor models to specify `F`.
    Specifically, the following two latent factor models are supported:
        - `pca`: Principal Component Analysis
        - `fa`: Factor Analysis

    Reference:
        [1] Fan, J., Liao, Y., & Liu, H. (2016). An overview of the estimation of large covariance and
            precision matrices. Econometrics Journal, 19(1), C1–C32. https://doi.org/10.1111/ectj.12061
        [2] Lin, H., Zhou, D., Liu, W., & Bian, J. (2021). Deep Risk Model: A Deep Learning Solution for
            Mining Latent Risk Factors to Improve Covariance Matrix Estimation. arXiv preprint arXiv:2107.05201.
    """

    FACTOR_MODEL_PCA = "pca"
    FACTOR_MODEL_FA = "fa"
    DEFAULT_NAN_OPTION = "fill"

    def __init__(self, factor_model: str = "pca", num_factors: int = 10, **kwargs):
        """
        Args:
            factor_model (str): the latent factor models used to estimate the structured covariance (`pca`/`fa`).
            num_factors (int): number of components to keep.
            kwargs: see `RiskModel` for more information
        """
        if "nan_option" in kwargs:
            assert kwargs["nan_option"] in [self.DEFAULT_NAN_OPTION], "nan_option={} is not supported".format(
                kwargs["nan_option"]
            )
        else:
            kwargs["nan_option"] = self.DEFAULT_NAN_OPTION

        super().__init__(**kwargs)

        assert factor_model in [
            self.FACTOR_MODEL_PCA,
            self.FACTOR_MODEL_FA,
        ], "factor_model={} is not supported".format(factor_model)
        self.solver = PCA if factor_model == self.FACTOR_MODEL_PCA else FactorAnalysis

        self.num_factors = num_factors

    def _predict(self, X: np.ndarray, return_decomposed_components=False) -> Union[np.ndarray, tuple]:
        """
        covariance estimation implementation

        Args:
            X (np.ndarray): data matrix containing multiple variables (columns) and observations (rows).
            return_decomposed_components (bool): whether return decomposed components of the covariance matrix.

        Returns:
            tuple or np.ndarray: decomposed covariance matrix or covariance matrix.
        """

        model = self.solver(self.num_factors, random_state=0).fit(X)

        F = model.components_.T  # variables x factors
        B = model.transform(X)  # observations x factors
        U = X - B @ F.T
        cov_b = np.cov(B.T)  # factors x factors
        var_u = np.var(U, axis=0)  # diagonal

        if return_decomposed_components:
            return F, cov_b, var_u

        cov_x = F @ cov_b @ F.T + np.diag(var_u)

        return cov_x
