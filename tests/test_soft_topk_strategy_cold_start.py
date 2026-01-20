import pandas as pd
import pytest

from qlib.contrib.strategy.cost_control import SoftTopkStrategy


class MockPosition:
    def __init__(self, weights):
        self.weights = weights

    def get_stock_weight_dict(self, only_stock=True):
        return self.weights


def create_test_strategy(topk, risk_degree, impact_limit):
    strat = SoftTopkStrategy.__new__(SoftTopkStrategy)
    strat.topk = topk
    strat.risk_degree = risk_degree
    strat.trade_impact_limit = impact_limit
    return strat


@pytest.mark.parametrize(
    ("impact_limit", "expected_fill"),
    [
        (0.1, 0.1),
        (1.0, 0.475),
    ],
)
def test_soft_topk_cold_start_impact_limit(impact_limit, expected_fill):
    scores = pd.Series({"C": 0.9, "D": 0.8, "A": 0.1, "B": 0.1})
    current_pos = MockPosition({})

    strat = create_test_strategy(topk=2, risk_degree=0.95, impact_limit=impact_limit)
    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    assert abs(res["C"] - expected_fill) < 1e-8
    assert abs(res["D"] - expected_fill) < 1e-8
