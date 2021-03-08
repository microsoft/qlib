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
        X = FB + U
    where `F` can be specified by explicit risk factors or latent factors.

    Therefore the structured covariance can be estimated by
        cov(X) = F cov(B) F.T + cov(U)

    We use latent factor models to estimate the structured covariance.
    Specifically, the following latent factor models are supported:
        - `pca`: Principal Component Analysis
        - `fa`: Factor Analysis

    Reference: [1] Fan, J., Liao, Y., & Liu, H. (2016). An overview of the estimation of large covariance and
    precision matrices. Econometrics Journal, 19(1), C1–C32. https://doi.org/10.1111/ectj.12061
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
        if "nan_option" in kwargs.keys():
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

        F = model.components_.T  # num_features x num_factors
        B = model.transform(X)  # num_samples x num_factors
        U = X - B @ F.T
        cov_b = np.cov(B.T)  # num_factors x num_factors
        var_u = np.var(U, axis=0)  # diagonal

        if return_decomposed_components:
            return F, cov_b, var_u

        cov_x = F @ cov_b @ F.T + np.diag(var_u)

        return cov_x
