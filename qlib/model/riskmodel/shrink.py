import numpy as np
from typing import Union

from qlib.model.riskmodel import RiskModel


class ShrinkCovEstimator(RiskModel):
    """Shrinkage Covariance Estimator

    This estimator will shrink the sample covariance matrix towards
    an identify matrix:
        S_hat = (1 - alpha) * S + alpha * F
    where `alpha` is the shrink parameter and `F` is the shrinking target.

    The following shrinking parameters (`alpha`) are supported:
        - `lw` [1][2][3]: use Ledoit-Wolf shrinking parameter.
        - `oas` [4]: use Oracle Approximating Shrinkage shrinking parameter.
        - float: directly specify the shrink parameter, should be between [0, 1].

    The following shrinking targets (`F`) are supported:
        - `const_var` [1][4][5]: assume stocks have the same constant variance and zero correlation.
        - `const_corr` [2][6]: assume stocks have different variance but equal correlation.
        - `single_factor` [3][7]: assume single factor model as the shrinking target.
        - np.ndarray: provide the shrinking targets directly.

    Note:
        - The optimal shrinking parameter depends on the selection of the shrinking target.
            Currently, `oas` is not supported for `const_corr` and `single_factor`.
        - Remember to set `nan_option` to `fill` or `mask` if your data has missing values.

    References:
        [1] Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
            Journal of Multivariate Analysis, 88(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4
        [2] Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix.
            Journal of Portfolio Management, 30(4), 1–22. https://doi.org/10.3905/jpm.2004.110
        [3] Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns
            with an application to portfolio selection.
            Journal of Empirical Finance, 10(5), 603–621. https://doi.org/10.1016/S0927-5398(03)00007-0
        [4] Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance
            estimation. IEEE Transactions on Signal Processing, 58(10), 5016–5029.
            https://doi.org/10.1109/TSP.2010.2053029
        [5] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-0000-00007f64e5b9/cov1para.m.zip
        [6] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffde5e2d4e/covCor.m.zip
        [7] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-0000-0000648dfc98/covMarket.m.zip
    """

    SHR_LW = "lw"
    SHR_OAS = "oas"

    TGT_CONST_VAR = "const_var"
    TGT_CONST_CORR = "const_corr"
    TGT_SINGLE_FACTOR = "single_factor"

    def __init__(self, alpha: Union[str, float] = 0.0, target: Union[str, np.ndarray] = "const_var", **kwargs):
        """
        Args:
            alpha (str or float): shrinking parameter or estimator (`lw`/`oas`)
            target (str or np.ndarray): shrinking target (`const_var`/`const_corr`/`single_factor`)
            kwargs: see `RiskModel` for more information
        """
        super().__init__(**kwargs)

        # alpha
        if isinstance(alpha, str):
            assert alpha in [self.SHR_LW, self.SHR_OAS], f"shrinking method `{alpha}` is not supported"
        elif isinstance(alpha, (float, np.floating)):
            assert 0 <= alpha <= 1, "alpha should be between [0, 1]"
        else:
            raise TypeError("invalid argument type for `alpha`")
        self.alpha = alpha

        # target
        if isinstance(target, str):
            assert target in [
                self.TGT_CONST_VAR,
                self.TGT_CONST_CORR,
                self.TGT_SINGLE_FACTOR,
            ], f"shrinking target `{target} is not supported"
        elif isinstance(target, np.ndarray):
            pass
        else:
            raise TypeError("invalid argument type for `target`")
        if alpha == self.SHR_OAS and target != self.TGT_CONST_VAR:
            raise NotImplementedError("currently `oas` can only support `const_var` as target")
        self.target = target

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # sample covariance
        S = super()._predict(X)

        # shrinking target
        F = self._get_shrink_target(X, S)

        # get shrinking parameter
        alpha = self._get_shrink_param(X, S, F)

        # shrink covariance
        if alpha > 0:
            S *= 1 - alpha
            F *= alpha
            S += F

        return S

    def _get_shrink_target(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """get shrinking target `F`"""
        if self.target == self.TGT_CONST_VAR:
            return self._get_shrink_target_const_var(X, S)
        if self.target == self.TGT_CONST_CORR:
            return self._get_shrink_target_const_corr(X, S)
        if self.target == self.TGT_SINGLE_FACTOR:
            return self._get_shrink_target_single_factor(X, S)
        return self.target

    def _get_shrink_target_const_var(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """get shrinking target with constant variance

        This target assumes zero pair-wise correlation and constant variance.
        The constant variance is estimated by averaging all sample's variances.
        """
        n = len(S)
        F = np.eye(n)
        np.fill_diagonal(F, np.mean(np.diag(S)))
        return F

    def _get_shrink_target_const_corr(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """get shrinking target with constant correlation

        This target assumes constant pair-wise correlation but keep the sample variance.
        The constant correlation is estimated by averaging all pairwise correlations.
        """
        n = len(S)
        var = np.diag(S)
        sqrt_var = np.sqrt(var)
        covar = np.outer(sqrt_var, sqrt_var)
        r_bar = (np.sum(S / covar) - n) / (n * (n - 1))
        F = r_bar * covar
        np.fill_diagonal(F, var)
        return F

    def _get_shrink_target_single_factor(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """get shrinking target with single factor model"""
        X_mkt = np.nanmean(X, axis=1)
        cov_mkt = np.asarray(X.T.dot(X_mkt) / len(X))
        var_mkt = np.asarray(X_mkt.dot(X_mkt) / len(X))
        F = np.outer(cov_mkt, cov_mkt) / var_mkt
        np.fill_diagonal(F, np.diag(S))
        return F

    def _get_shrink_param(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """get shrinking parameter `alpha`

        Note:
            The Ledoit-Wolf shrinking parameter estimator consists of three different methods.
        """
        if self.alpha == self.SHR_OAS:
            return self._get_shrink_param_oas(X, S, F)
        elif self.alpha == self.SHR_LW:
            if self.target == self.TGT_CONST_VAR:
                return self._get_shrink_param_lw_const_var(X, S, F)
            if self.target == self.TGT_CONST_CORR:
                return self._get_shrink_param_lw_const_corr(X, S, F)
            if self.target == self.TGT_SINGLE_FACTOR:
                return self._get_shrink_param_lw_single_factor(X, S, F)
        return self.alpha

    def _get_shrink_param_oas(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """Oracle Approximating Shrinkage Estimator

        This method uses the following formula to estimate the `alpha`
        parameter for the shrink covariance estimator:
            A = (1 - 2 / p) * trace(S^2) + trace^2(S)
            B = (n + 1 - 2 / p) * (trace(S^2) - trace^2(S) / p)
            alpha = A / B
        where `n`, `p` are the dim of observations and variables respectively.
        """
        trS2 = np.sum(S ** 2)
        tr2S = np.trace(S) ** 2

        n, p = X.shape

        A = (1 - 2 / p) * (trS2 + tr2S)
        B = (n + 1 - 2 / p) * (trS2 + tr2S / p)
        alpha = A / B

        return alpha

    def _get_shrink_param_lw_const_var(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """Ledoit-Wolf Shrinkage Estimator (Constant Variance)

        This method shrinks the covariance matrix towards the constand variance target.
        """
        t, n = X.shape

        y = X ** 2
        phi = np.sum(y.T.dot(y) / t - S ** 2)

        gamma = np.linalg.norm(S - F, "fro") ** 2

        kappa = phi / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha

    def _get_shrink_param_lw_const_corr(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """Ledoit-Wolf Shrinkage Estimator (Constant Correlation)

        This method shrinks the covariance matrix towards the constand correlation target.
        """
        t, n = X.shape

        var = np.diag(S)
        sqrt_var = np.sqrt(var)
        r_bar = (np.sum(S / np.outer(sqrt_var, sqrt_var)) - n) / (n * (n - 1))

        y = X ** 2
        phi_mat = y.T.dot(y) / t - S ** 2
        phi = np.sum(phi_mat)

        theta_mat = (X ** 3).T.dot(X) / t - var[:, None] * S
        np.fill_diagonal(theta_mat, 0)
        rho = np.sum(np.diag(phi_mat)) + r_bar * np.sum(np.outer(1 / sqrt_var, sqrt_var) * theta_mat)

        gamma = np.linalg.norm(S - F, "fro") ** 2

        kappa = (phi - rho) / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha

    def _get_shrink_param_lw_single_factor(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """Ledoit-Wolf Shrinkage Estimator (Single Factor Model)

        This method shrinks the covariance matrix towards the single factor model target.
        """
        t, n = X.shape

        X_mkt = np.nanmean(X, axis=1)
        cov_mkt = np.asarray(X.T.dot(X_mkt) / len(X))
        var_mkt = np.asarray(X_mkt.dot(X_mkt) / len(X))

        y = X ** 2
        phi = np.sum(y.T.dot(y)) / t - np.sum(S ** 2)

        rdiag = np.sum(y ** 2) / t - np.sum(np.diag(S) ** 2)
        z = X * X_mkt[:, None]
        v1 = y.T.dot(z) / t - cov_mkt[:, None] * S
        roff1 = np.sum(v1 * cov_mkt[:, None].T) / var_mkt - np.sum(np.diag(v1) * cov_mkt) / var_mkt
        v3 = z.T.dot(z) / t - var_mkt * S
        roff3 = (
            np.sum(v3 * np.outer(cov_mkt, cov_mkt)) / var_mkt ** 2 - np.sum(np.diag(v3) * cov_mkt ** 2) / var_mkt ** 2
        )
        roff = 2 * roff1 - roff3
        rho = rdiag + roff

        gamma = np.linalg.norm(S - F, "fro") ** 2

        kappa = (phi - rho) / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha
