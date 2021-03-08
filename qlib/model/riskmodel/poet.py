import numpy as np

from qlib.model.riskmodel import RiskModel


class POETCovEstimator(RiskModel):
    """Principal Orthogonal Complement Thresholding Estimator (POET)

    Reference:
        [1] Fan, J., Liao, Y., & Mincheva, M. (2013). Large covariance estimation by thresholding principal orthogonal complements.
            Journal of the Royal Statistical Society. Series B: Statistical Methodology, 75(4), 603–680. https://doi.org/10.1111/rssb.12016
        [2] http://econweb.rutgers.edu/yl1114/papers/poet/POET.m
    """

    THRESH_SOFT = "soft"
    THRESH_HARD = "hard"
    THRESH_SCAD = "scad"

    def __init__(self, num_factors: int = 0, thresh: float = 1.0, thresh_method: str = "soft", **kwargs):
        """
        Args:
            num_factors (int): number of factors (if set to zero, no factor model will be used).
            thresh (float): the positive constant for thresholding.
            thresh_method (str): thresholding method, which can be
                - 'soft': soft thresholding.
                - 'hard': hard thresholding.
                - 'scad': scad thresholding.
            kwargs: see `RiskModel` for more information.
        """
        super().__init__(**kwargs)

        assert num_factors >= 0, "`num_factors` requires a positive integer"
        self.num_factors = num_factors

        assert thresh >= 0, "`thresh` requires a positive float number"
        self.thresh = thresh

        assert thresh_method in [
            self.THRESH_HARD,
            self.THRESH_SOFT,
            self.THRESH_SCAD,
        ], "`thresh_method` should be `soft`/`hard`/`scad`"
        self.thresh_method = thresh_method

    def _predict(self, X: np.ndarray) -> np.ndarray:

        Y = X.T  # NOTE: to match POET's implementation
        p, n = Y.shape

        if self.num_factors > 0:
            Dd, V = np.linalg.eig(Y.T.dot(Y))
            V = V[:, np.argsort(Dd)]
            F = V[:, -self.num_factors :][:, ::-1] * np.sqrt(n)
            LamPCA = Y.dot(F) / n
            uhat = np.asarray(Y - LamPCA.dot(F.T))
            Lowrank = np.asarray(LamPCA.dot(LamPCA.T))
            rate = 1 / np.sqrt(p) + np.sqrt(np.log(p) / n)
        else:
            uhat = np.asarray(Y)
            rate = np.sqrt(np.log(p) / n)
            Lowrank = 0

        lamb = rate * self.thresh
        SuPCA = uhat.dot(uhat.T) / n
        SuDiag = np.diag(np.diag(SuPCA))
        R = np.linalg.inv(SuDiag ** 0.5).dot(SuPCA).dot(np.linalg.inv(SuDiag ** 0.5))

        if self.thresh_method == self.THRESH_HARD:
            M = R * (np.abs(R) > lamb)
        elif self.thresh_method == self.THRESH_SOFT:
            res = np.abs(R) - lamb
            res = (res + np.abs(res)) / 2
            M = np.sign(R) * res
        else:
            M1 = (np.abs(R) < 2 * lamb) * np.sign(R) * (np.abs(R) - lamb) * (np.abs(R) > lamb)
            M2 = (np.abs(R) < 3.7 * lamb) * (np.abs(R) >= 2 * lamb) * (2.7 * R - 3.7 * np.sign(R) * lamb) / 1.7
            M3 = (np.abs(R) >= 3.7 * lamb) * R
            M = M1 + M2 + M3

        Rthresh = M - np.diag(np.diag(M)) + np.eye(p)
        SigmaU = (SuDiag ** 0.5).dot(Rthresh).dot(SuDiag ** 0.5)
        SigmaY = SigmaU + Lowrank

        return SigmaY
