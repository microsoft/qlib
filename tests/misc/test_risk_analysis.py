import unittest

import numpy as np
import pandas as pd

from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.evaluate_portfolio import get_max_drawdown_from_series
from qlib.contrib.report.analysis_position.report import _calculate_mdd


def _series(values):
    return pd.Series(values, index=pd.date_range("2024-01-02", periods=len(values), freq="B"))


class TestMaxDrawdown(unittest.TestCase):
    def test_loss_in_first_period_is_a_drawdown(self):
        r = _series([-0.10, 0.05, 0.06, 0.02, 0.01])
        for mode in ("sum", "product"):
            res = risk_analysis(r, N=252, mode=mode).loc["max_drawdown", "risk"]
            self.assertAlmostEqual(res, -0.10, places=10, msg=mode)

    def test_losses_before_the_first_high(self):
        r = _series([-0.05, -0.05, 0.03, 0.04, 0.02])
        self.assertAlmostEqual(risk_analysis(r, N=252, mode="sum").loc["max_drawdown", "risk"], -0.10, places=10)
        self.assertAlmostEqual(risk_analysis(r, N=252, mode="product").loc["max_drawdown", "risk"], -0.0975, places=10)
        self.assertAlmostEqual(get_max_drawdown_from_series(r), -0.0975, places=10)
        np.testing.assert_allclose(_calculate_mdd(r.cumsum()).values, [-0.05, -0.10, -0.07, -0.03, -0.01])

    def test_gain_first_is_unchanged(self):
        r = _series([0.02, -0.05, 0.03, 0.04, 0.02])
        for mode in ("sum", "product"):
            res = risk_analysis(r, N=252, mode=mode).loc["max_drawdown", "risk"]
            self.assertAlmostEqual(res, -0.05, places=10, msg=mode)


if __name__ == "__main__":
    unittest.main()
