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
    NAN_OPTION = "fill"

    def __init__(
            self,
            factor_model: str = "pca",
            num_factors: int = 10,
            assume_centered: bool = False,
            scale_return: bool = True,
    ):
        """
        Args:
            factor_model (str): the latent factor models used to estimate the structured covariance (`pca`/`fa`).
            num_factors (int): number of components to keep.
            assume_centered (bool): whether the data is assumed to be centered.
            scale_return (bool): whether scale returns as percentage.
        """
        super().__init__(self.NAN_OPTION, assume_centered, scale_return)

        assert factor_model in [
            self.FACTOR_MODEL_PCA,
            self.FACTOR_MODEL_FA,
        ], "factor_model={} is not supported".format(factor_model)
        self.solver = PCA if factor_model == self.FACTOR_MODEL_PCA else FactorAnalysis

        self.num_factors = num_factors

    def predict(
            self,
            X: Union[pd.Series, pd.DataFrame, np.ndarray],
            return_corr: bool = False,
            is_price: bool = True,
            return_decomposed_components=False,
    ) -> Union[pd.DataFrame, np.ndarray, tuple]:
        """
        Args:
            X (pd.Series, pd.DataFrame or np.ndarray): data from which to estimate the covariance,
                with variables as columns and observations as rows.
            return_corr (bool): whether return the correlation matrix.
            is_price (bool): whether `X` contains price (if not assume stock returns).
            return_decomposed_components (bool): whether return decomposed components of the covariance matrix.

        Returns:
            tuple or pd.DataFrame or np.ndarray: decomposed covariance matrix or estimated covariance or correlation.
        """
        assert (
                not return_corr or not return_decomposed_components
        ), "Can only return either correlation matrix or decomposed components."

        # transform input into 2D array
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            columns = None
        else:
            if isinstance(X.index, pd.MultiIndex):
                if isinstance(X, pd.DataFrame):
                    X = X.iloc[:, 0].unstack(level="instrument")  # always use the first column
                else:
                    X = X.unstack(level="instrument")
            else:
                # X is 2D DataFrame
                pass
            columns = X.columns  # will be used to restore dataframe
            X = X.values

        # calculate pct_change
        if is_price:
            X = X[1:] / X[:-1] - 1  # NOTE: resulting `n - 1` rows

        # scale return
        if self.scale_return:
            X *= 100

        # handle nan and centered
        X = self._preprocess(X)

        if return_decomposed_components:
            F, cov_b, var_u = self._predict(X, return_structured=True)
            return F, cov_b, var_u
        else:
            # estimate covariance
            S = self._predict(X)

            # return correlation if needed
            if return_corr:
                vola = np.sqrt(np.diag(S))
                corr = S / np.outer(vola, vola)
                if columns is None:
                    return corr
                return pd.DataFrame(corr, index=columns, columns=columns)

            # return covariance
            if columns is None:
                return S
            return pd.DataFrame(S, index=columns, columns=columns)

    def _predict(self, X: np.ndarray, return_structured=False) -> Union[np.ndarray, tuple]:
        """
        covariance estimation implementation

        Args:
            X (np.ndarray): data matrix containing multiple variables (columns) and observations (rows).
            return_structured (bool): whether return decomposed components of the covariance matrix.

        Returns:
            tuple or np.ndarray: decomposed covariance matrix or covariance matrix.
        """

        model = self.solver(self.num_factors, random_state=0).fit(X)

        F = model.components_.T  # num_features x num_factors
        B = model.transform(X)  # num_samples x num_factors
        U = X - B @ F.T
        cov_b = np.cov(B.T)  # num_factors x num_factors
        var_u = np.var(U, axis=0)  # diagonal

        if return_structured:
            return F, cov_b, var_u

        cov_x = F @ cov_b @ F.T + np.diag(var_u)

        return cov_x
