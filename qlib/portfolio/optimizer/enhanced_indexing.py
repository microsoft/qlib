# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import cvxpy as cp
import pandas as pd
from typing import Union

from qlib.portfolio.optimizer import BaseOptimizer


class EnhancedIndexingOptimizer(BaseOptimizer):
    """
    Portfolio Optimizer with Enhanced Indexing

    Note:
        This optimizer always assumes full investment and no-shorting.
    """

    START_FROM_W0 = "w0"
    START_FROM_BENCH = "benchmark"

    def __init__(
        self,
        lamb: float = 10,
        delta: float = 0.4,
        bench_dev: float = 0.01,
        inds_dev: float = None,
        scale_alpha: bool = True,
        verbose: bool = False,
        warm_start: str = None,
        max_iters: int = 10000,
    ):
        """
        Args:
            lamb (float): risk aversion parameter (larger `lamb` means less focus on return)
            delta (float): turnover rate limit
            bench_dev (float): benchmark deviation limit
            inds_dev (float/None): industry deviation limit, set `inds_dev` to None to ignore industry specific
                                   restriction
            scale_alpha (bool): if to scale alpha to match the volatility of the covariance matrix
            verbose (bool): if print detailed information about the solver
            warm_start (str): whether try to warm start (`w0`/`benchmark`/``)
                              (https://www.cvxpy.org/tutorial/advanced/index.html#warm-start)
        """

        assert lamb >= 0, "risk aversion parameter `lamb` should be positive"
        self.lamb = lamb

        assert delta >= 0, "turnover limit `delta` should be positive"
        self.delta = delta

        assert bench_dev >= 0, "benchmark deviation limit `bench_dev` should be positive"
        self.bench_dev = bench_dev

        assert inds_dev is None or inds_dev >= 0, "industry deviation limit `inds_dev` should be positive or None."
        self.inds_dev = inds_dev

        assert warm_start in [
            None,
            self.START_FROM_W0,
            self.START_FROM_BENCH,
        ], "illegal warm start option"
        self.start_from_w0 = warm_start == self.START_FROM_W0
        self.start_from_bench = warm_start == self.START_FROM_BENCH

        self.scale_alpha = scale_alpha
        self.verbose = verbose
        self.max_iters = max_iters

    def __call__(
        self,
        u: Union[np.ndarray, pd.Series],
        F: np.ndarray,
        covB: np.ndarray,
        varU: np.ndarray,
        w0: np.ndarray,
        w_bench: np.ndarray,
        inds_onehot: np.ndarray = None,
    ) -> Union[np.ndarray, pd.Series]:
        """
        Args:
            u (np.ndarray or pd.Series): expected returns (a.k.a., alpha)
            F, covB, varU (np.ndarray): see StructuredCovEstimator
            w0 (np.ndarray): initial weights (for turnover control)
            w_bench (np.ndarray): benchmark weights
            inds_onehot (np.ndarray): industry (onehot)

        Returns:
            np.ndarray or pd.Series: optimized portfolio allocation
        """
        assert inds_onehot is not None or self.inds_dev is None, "Industry onehot vector is required."

        # transform dataframe into array
        if isinstance(u, pd.Series):
            u = u.values

        # scale alpha to match volatility
        if self.scale_alpha:
            u = u / u.std()
            x_variance = np.mean(np.diag(F @ covB @ F.T) + varU)
            u *= x_variance ** 0.5

        w = cp.Variable(len(u))  # num_assets
        v = w @ F  # num_factors
        ret = w @ u
        risk = cp.quad_form(v, covB) + cp.sum(cp.multiply(varU, w ** 2))
        obj = cp.Maximize(ret - self.lamb * risk)
        d_bench = w - w_bench
        cons = [
            w >= 0,
            cp.sum(w) == 1,
            d_bench >= -self.bench_dev,
            d_bench <= self.bench_dev,
        ]

        if self.inds_dev is not None:
            d_inds = d_bench @ inds_onehot
            cons.append(d_inds >= -self.inds_dev)
            cons.append(d_inds <= self.inds_dev)

        if w0 is not None:
            turnover = cp.sum(cp.abs(w - w0))
            cons.append(turnover <= self.delta)

        warm_start = False
        if self.start_from_w0:
            if w0 is None:
                print("Warning: try warm start with w0, but w0 is `None`.")
            else:
                w.value = w0
                warm_start = True
        elif self.start_from_bench:
            w.value = w_bench
            warm_start = True

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.SCS, verbose=self.verbose, warm_start=warm_start, max_iters=self.max_iters)

        if prob.status != "optimal":
            print("Warning: solve failed.", prob.status)

        return np.asarray(w.value)
