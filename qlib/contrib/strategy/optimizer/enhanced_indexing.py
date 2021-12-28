# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import cvxpy as cp
import pandas as pd

from typing import Union, Optional, Dict, Any, List

from qlib.log import get_module_logger
from .base import BaseOptimizer


logger = get_module_logger("EnhancedIndexingOptimizer")


class EnhancedIndexingOptimizer(BaseOptimizer):
    """
    Portfolio Optimizer for Enhanced Indexing

    Notations:
        w0: current holding weights
        wb: benchmark weight
        r: expected return
        F: factor exposure
        cov_b: factor covariance
        var_u: residual variance (diagonal)
        lamb: risk aversion parameter
        delta: total turnover limit
        b_dev: benchmark deviation limit
        f_dev: factor deviation limit

    Also denote:
        d = w - wb: benchmark deviation
        v = d @ F: factor deviation

    The optimization problem for enhanced indexing:
        max_w  d @ r - lamb * (v @ cov_b @ v + var_u @ d**2)
        s.t.   w >= 0
               sum(w) == 1
               sum(|w - w0|) <= delta
               d >= -b_dev
               d <= b_dev
               v >= -f_dev
               v <= f_dev
    """

    def __init__(
        self,
        lamb: float = 1,
        delta: Optional[float] = 0.2,
        b_dev: Optional[float] = 0.01,
        f_dev: Optional[Union[List[float], np.ndarray]] = None,
        scale_return: bool = True,
        epsilon: float = 5e-5,
        solver_kwargs: Optional[Dict[str, Any]] = {},
    ):
        """
        Args:
            lamb (float): risk aversion parameter (larger `lamb` means more focus on risk)
            delta (float): total turnover limit
            b_dev (float): benchmark deviation limit
            f_dev (list): factor deviation limit
            scale_return (bool): whether scale return to match estimated volatility
            epsilon (float): minimum weight
            solver_kwargs (dict): kwargs for cvxpy solver
        """

        assert lamb >= 0, "risk aversion parameter `lamb` should be positive"
        self.lamb = lamb

        assert delta >= 0, "turnover limit `delta` should be positive"
        self.delta = delta

        assert b_dev is None or b_dev >= 0, "benchmark deviation limit `b_dev` should be positive"
        self.b_dev = b_dev

        if isinstance(f_dev, float):
            assert f_dev >= 0, "factor deviation limit `f_dev` should be positive"
        elif f_dev is not None:
            f_dev = np.array(f_dev)
            assert all(f_dev >= 0), "factor deviation limit `f_dev` should be positive"
        self.f_dev = f_dev

        self.scale_return = scale_return
        self.epsilon = epsilon
        self.solver_kwargs = solver_kwargs

    def __call__(
        self,
        r: np.ndarray,
        F: np.ndarray,
        cov_b: np.ndarray,
        var_u: np.ndarray,
        w0: np.ndarray,
        wb: np.ndarray,
        mfh: Optional[np.ndarray] = None,
        mfs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            r (np.ndarray): expected returns
            F (np.ndarray): factor exposure
            cov_b (np.ndarray): factor covariance
            var_u (np.ndarray): residual variance
            w0 (np.ndarray): current holding weights
            wb (np.ndarray): benchmark weights
            mfh (np.ndarray): mask force holding
            mfs (np.ndarray): mask force selling

        Returns:
            np.ndarray: optimized portfolio allocation
        """
        # scale return to match volatility
        if self.scale_return:
            r = r / r.std()
            r *= np.sqrt(np.mean(np.diag(F @ cov_b @ F.T) + var_u))

        # target weight
        w = cp.Variable(len(r), nonneg=True)
        w.value = wb  # for warm start

        # precompute exposure
        d = w - wb  # benchmark exposure
        v = d @ F  # factor exposure

        # objective
        ret = d @ r  # excess return
        risk = cp.quad_form(v, cov_b) + var_u @ (d ** 2)  # tracking error
        obj = cp.Maximize(ret - self.lamb * risk)

        # weight bounds
        lb = np.zeros_like(wb)
        ub = np.ones_like(wb)

        # bench bounds
        if self.b_dev is not None:
            lb = np.maximum(lb, wb - self.b_dev)
            ub = np.minimum(ub, wb + self.b_dev)

        # force holding
        if mfh is not None:
            lb[mfh] = w0[mfh]
            ub[mfh] = w0[mfh]

        # force selling
        # NOTE: this will override mfh
        if mfs is not None:
            lb[mfs] = 0
            ub[mfs] = 0

        # constraints
        # TODO: currently we assume fullly invest in the stocks,
        # in the future we should support holding cash as an asset
        cons = [cp.sum(w) == 1, w >= lb, w <= ub]

        # factor deviation
        if self.f_dev is not None:
            cons.extend([v >= -self.f_dev, v <= self.f_dev])

        # total turnover constraint
        t_cons = []
        if self.delta is not None:
            if w0 is not None and w0.sum() > 0:
                t_cons.extend([cp.norm(w - w0, 1) <= self.delta])

        # optimize
        # trial 1: use all constraints
        success = False
        try:
            prob = cp.Problem(obj, cons + t_cons)
            prob.solve(solver=cp.ECOS, warm_start=True, **self.solver_kwargs)
            assert prob.status == "optimal"
            success = True
        except Exception as e:
            logger.warning(f"trial 1 failed {e} (status: {prob.status})")

        # trial 2: remove turnover constraint
        if not success and len(t_cons):
            logger.info("try removing turnover constraint as the last optimization failed")
            try:
                w.value = wb
                prob = cp.Problem(obj, cons)
                prob.solve(solver=cp.ECOS, warm_start=True, **self.solver_kwargs)
                assert prob.status in ["optimal", "optimal_inaccurate"]
                success = True
            except Exception as e:
                logger.warning(f"trial 2 failed {e} (status: {prob.status})")

        # return current weight if not success
        if not success:
            logger.warning("optimization failed, will return current holding weight")
            return w0

        if prob.status == "optimal_inaccurate":
            logger.warning(f"the optimization is inaccurate")

        # remove small weight
        w = np.asarray(w.value)
        w[w < self.epsilon] = 0
        w /= w.sum()

        return w
