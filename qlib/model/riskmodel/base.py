# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
import numpy as np
import pandas as pd
from typing import Union

from qlib.model.base import BaseModel


class RiskModel(BaseModel):
    """Risk Model

    A risk model is used to estimate the covariance matrix of stock returns.
    """

    MASK_NAN = "mask"
    FILL_NAN = "fill"
    IGNORE_NAN = "ignore"

    def __init__(self, nan_option: str = "ignore", assume_centered: bool = False, scale_return: bool = True):
        """
        Args:
            nan_option (str): nan handling option (`ignore`/`mask`/`fill`).
            assume_centered (bool): whether the data is assumed to be centered.
            scale_return (bool): whether scale returns as percentage.
        """
        # nan
        assert nan_option in [
            self.MASK_NAN,
            self.FILL_NAN,
            self.IGNORE_NAN,
        ], f"`nan_option={nan_option}` is not supported"
        self.nan_option = nan_option

        self.assume_centered = assume_centered
        self.scale_return = scale_return

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
            pd.DataFrame or np.ndarray: estimated covariance (or correlation).
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

        # return decomposed components if needed
        if return_decomposed_components:
            assert (
                "return_decomposed_components" in inspect.getfullargspec(self._predict).args
            ), "This risk model does not support return decomposed components of the covariance matrix "

            F, cov_b, var_u = self._predict(X, return_decomposed_components=True)
            return F, cov_b, var_u

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

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """covariance estimation implementation

        This method should be overridden by child classes.

        By default, this method implements the empirical covariance estimation.

        Args:
            X (np.ndarray): data matrix containing multiple variables (columns) and observations (rows).

        Returns:
            np.ndarray: covariance matrix.
        """
        xTx = np.asarray(X.T.dot(X))
        N = len(X)
        if isinstance(X, np.ma.MaskedArray):
            M = 1 - X.mask
            N = M.T.dot(M)  # each pair has distinct number of samples
        return xTx / N

    def _preprocess(self, X: np.ndarray) -> Union[np.ndarray, np.ma.MaskedArray]:
        """handle nan and centerize data

        Note:
            if `nan_option='mask'` then the returned array will be `np.ma.MaskedArray`.
        """
        # handle nan
        if self.nan_option == self.FILL_NAN:
            X = np.nan_to_num(X)
        elif self.nan_option == self.MASK_NAN:
            X = np.ma.masked_invalid(X)
        # centralize
        if not self.assume_centered:
            X = X - np.nanmean(X, axis=0)
        return X
