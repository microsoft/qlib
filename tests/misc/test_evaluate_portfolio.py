# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

from qlib.contrib.evaluate_portfolio import get_alpha, get_beta


@pytest.mark.parametrize("length", [2, 5, 50])
def test_beta_of_benchmark_is_one(length):
    benchmark = pd.Series(np.linspace(-0.01, 0.02, length))

    beta = get_beta(benchmark, benchmark)

    assert np.isscalar(beta)
    assert beta == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [-1.0, 0.0, 0.5, 2.0])
def test_beta_of_affine_benchmark(scale):
    benchmark = pd.Series([-0.003, -0.001, 0.002, 0.004, 0.001])
    returns = scale * benchmark + 0.001

    beta = get_beta(returns, benchmark)

    assert np.isscalar(beta)
    assert beta == pytest.approx(scale)


def test_beta_matches_regression_slope():
    benchmark = pd.Series([-0.01, 0.02, 0.01, -0.02, 0.03])
    returns = pd.Series([0.015, 0.025, 0.035, -0.025, 0.04])
    expected = linregress(benchmark, returns).slope

    beta = get_beta(returns, benchmark)

    assert np.isscalar(beta)
    assert beta == pytest.approx(expected)


@pytest.mark.parametrize("risk_free_rate", [0.0, 0.03])
def test_alpha_of_benchmark_is_zero(risk_free_rate):
    benchmark = pd.Series([-0.003, -0.001, 0.002, 0.004, 0.001])

    alpha = get_alpha(benchmark, benchmark, risk_free_rate=risk_free_rate)

    assert np.isscalar(alpha)
    assert alpha == pytest.approx(0.0, abs=1e-12)
