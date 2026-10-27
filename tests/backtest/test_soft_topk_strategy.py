import pandas as pd
import pytest
from qlib.contrib.strategy.cost_control import SoftTopkStrategy


class MockPosition:
    def __init__(self, weights):
        self.weights = weights

    def get_stock_weight_dict(self, only_stock=True):
        return self.weights


def test_soft_topk_logic():
    # Initial: A=0.8, B=0.2 (Total=1.0). Target Risk=0.95.
    # Scores: A and B are low, C and D are topk.
    scores = pd.Series({"C": 0.9, "D": 0.8, "A": 0.1, "B": 0.1})
    current_pos = MockPosition({"A": 0.8, "B": 0.2})

    topk = 2
    risk_degree = 0.95
    impact_limit = 0.1  # Max change per step

    def create_test_strategy(impact_limit_value):
        strat = SoftTopkStrategy.__new__(SoftTopkStrategy)
        strat.topk = topk
        strat.risk_degree = risk_degree
        strat.trade_impact_limit = impact_limit_value
        return strat

    # 1. With impact limit: Expect deterministic sell and limited buy
    strat_i = create_test_strategy(impact_limit)
    res_i = strat_i.generate_target_weight_position(scores, current_pos, None, None)

    # A should be exactly 0.8 - 0.1 = 0.7
    assert abs(res_i["A"] - 0.7) < 1e-8
    # B should be exactly 0.2 - 0.1 = 0.1
    assert abs(res_i["B"] - 0.1) < 1e-8
    # Total sells = 0.2 released. New budget = 0.2 + (0.95 - 1.0) = 0.15.
    # C and D share 0.15 -> 0.075 each.
    assert abs(res_i["C"] - 0.075) < 1e-8
    assert abs(res_i["D"] - 0.075) < 1e-8

    # 2. Without impact limit: Expect full liquidation and full target fill
    strat_c = create_test_strategy(1.0)
    res_c = strat_c.generate_target_weight_position(scores, current_pos, None, None)

    # A, B not in topk -> Liquidated
    assert "A" not in res_c and "B" not in res_c
    # C, D should reach ideal_per_stock (0.95/2 = 0.475)
    assert abs(res_c["C"] - 0.475) < 1e-8
    assert abs(res_c["D"] - 0.475) < 1e-8


def test_soft_topk_target_weights_follow_get_risk_degree_override():
    scores = pd.Series({"C": 0.9, "D": 0.8, "A": 0.1, "B": 0.1})
    current_pos = MockPosition({})

    class HalfRisk(SoftTopkStrategy):
        def get_risk_degree(self, trade_step=None):
            return 0.5

    strat = HalfRisk.__new__(HalfRisk)
    strat.topk = 2
    strat.risk_degree = 0.95
    strat.trade_impact_limit = 1.0

    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    assert abs(res["C"] - 0.25) < 1e-8
    assert abs(res["D"] - 0.25) < 1e-8


def test_soft_topk_passes_trade_step_into_get_risk_degree():
    scores = pd.Series({"C": 0.9, "D": 0.8, "A": 0.1, "B": 0.1})
    current_pos = MockPosition({})
    seen = []

    class StepRisk(SoftTopkStrategy):
        def get_risk_degree(self, trade_step=None):
            seen.append(trade_step)
            if trade_step == 0:
                return 0.5
            return 0.9

    class StepCalendar:
        def get_trade_step(self):
            return 3

    strat = StepRisk.__new__(StepRisk)
    strat.topk = 2
    strat.risk_degree = 0.95
    strat.trade_impact_limit = 1.0
    strat.level_infra = {"trade_calendar": StepCalendar()}

    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    assert seen == [3]
    assert abs(res["C"] - 0.45) < 1e-8
    assert abs(res["D"] - 0.45) < 1e-8


# Sentinel for a calendar whose step attribute was never set.
_UNINITIALIZED_STEP = object()


class _CalendarStep:
    """Minimal trade calendar. An uninitialized step raises, as a missing attribute would."""

    def __init__(self, step):
        self._step = step

    def get_trade_step(self):
        if self._step is _UNINITIALIZED_STEP:
            raise AttributeError("trade_step")
        return self._step


@pytest.mark.parametrize(
    ("step", "expected_step"),
    [
        ("missing", None),
        (None, None),
        (-1, None),
        (_UNINITIALIZED_STEP, None),
        (0, 0),
    ],
)
def test_soft_topk_unstarted_calendar_does_not_forward_step(step, expected_step):
    # level_infra is set, but the calendar is missing, has not started (step < 0),
    # or has no step yet. A getter that indexes by step must see None — the old
    # no-argument call — and must not raise. Step 0 is the first bar and is passed through.
    scores = pd.Series({"C": 0.9, "D": 0.8, "A": 0.1, "B": 0.1})
    current_pos = MockPosition({})
    seen = []

    class StepRisk(SoftTopkStrategy):
        def get_risk_degree(self, trade_step=None):
            seen.append(trade_step)
            if trade_step is None:
                return 0.5
            if type(trade_step) is not int or trade_step < 0:
                raise ValueError(f"calendar has not started: {trade_step!r}")
            return 0.4

    strat = StepRisk.__new__(StepRisk)
    strat.topk = 2
    strat.risk_degree = 0.95
    strat.trade_impact_limit = 1.0
    calendar = None if step == "missing" else _CalendarStep(step)
    strat.level_infra = {"trade_calendar": calendar}

    res = strat.generate_target_weight_position(scores, current_pos, None, None)

    expected_weight = (0.4 if expected_step == 0 else 0.5) / 2
    assert seen == [expected_step]
    assert abs(res["C"] - expected_weight) < 1e-8
    assert abs(res["D"] - expected_weight) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__])
