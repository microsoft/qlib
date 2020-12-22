#  Copyright (c) Qingyao Sun.
#  Licensed under the MIT License.

import unittest

import numpy as np
from cvxopt import matrix, solvers

from qlib.portfolio.optimizer import PortfolioOptimizer


class TestOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        print("In method", self._testMethodName)

        np.random.seed(42)
        self.S = np.cov(np.random.random((50, 1024)))
        self.u = np.random.random(50)
        self.w0 = np.random.random(50)
        self.w0 /= self.w0.sum()

    def test_gmv(self):
        delta = np.random.random()
        alpha = np.random.random()

        optimizer = PortfolioOptimizer("gmv", delta=delta, alpha=alpha)
        w_qlib = optimizer(self.S, w0=self.w0)

        upper_bound = np.minimum(1.0, self.w0 + delta)
        lower_bound = np.maximum(0.0, self.w0 - delta)
        identity = np.diag(np.repeat(1.0, self.w0.size))
        P = alpha * identity + self.S
        q = np.zeros_like(self.w0)
        G = np.vstack((identity, -identity))
        h = np.concatenate((upper_bound, -lower_bound))
        A = np.expand_dims(np.ones_like(self.w0), 0)
        b = np.array([1.0])

        solvers.options["show_progress"] = False
        sol = solvers.qp(
            matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b), initvals={"x": matrix(self.w0)}
        )
        self.assertEqual(sol["status"], "optimal")
        w_cvxopt = np.array(sol["x"]).squeeze()

        for algo, w in [("Initial", self.w0), ("Qlib", w_qlib), ("CVXOPT", w_cvxopt)]:
            self.assertTrue(np.all(w <= upper_bound), msg=algo)
            self.assertTrue(np.all(lower_bound <= w), msg=algo)
            self.assertAlmostEqual(w.sum(), 1.0, msg=algo)
            loss = alpha * w.T @ w + w.T @ self.S @ w
            print(f"{algo=}, {loss=}")

    def test_mvo(self):
        lamb = np.random.random()
        delta = np.random.random()
        alpha = np.random.random()

        optimizer = PortfolioOptimizer("mvo", lamb=lamb, delta=delta, alpha=alpha)
        w_qlib = optimizer(self.S, self.u, self.w0)

        upper_bound = np.minimum(1.0, self.w0 + delta)
        lower_bound = np.maximum(0.0, self.w0 - delta)
        identity = np.diag(np.repeat(1.0, self.w0.size))
        P = 2 * (alpha * identity + self.S)
        q = -self.u
        G = np.vstack((identity, -identity))
        h = np.concatenate((upper_bound, -lower_bound))
        A = np.expand_dims(np.ones_like(self.w0), 0)
        b = np.array([1.0])

        solvers.options["show_progress"] = False
        sol = solvers.qp(
            matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b), initvals={"x": matrix(self.w0)}
        )
        self.assertEqual(sol["status"], "optimal")
        w_cvxopt = np.array(sol["x"]).squeeze()

        for algo, w in [("Initial", self.w0), ("Qlib", w_qlib), ("CVXOPT", w_cvxopt)]:
            self.assertTrue(np.all(G @ w <= h))
            self.assertTrue(np.allclose(A @ w, b))

            self.assertTrue(np.all(w <= upper_bound), msg=algo)
            self.assertTrue(np.all(lower_bound <= w), msg=algo)
            self.assertAlmostEqual(w.sum(), 1.0, msg=algo)
            loss = alpha * w.T @ w - w.T @ self.u + lamb * w.T @ self.S @ w
            print(f"{algo=}, {loss=}")


if __name__ == "__main__":
    unittest.main()
