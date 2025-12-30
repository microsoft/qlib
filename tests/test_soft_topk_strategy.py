import pandas as pd
import pytest
from qlib.contrib.strategy.cost_control import SoftTopkStrategy

class MockPosition:
    def __init__(self, weights): self.weights = weights
    def get_stock_weight_dict(self, only_stock=True): return self.weights

def test_soft_topk_logic():
    # Initial: A=0.8, B=0.2 (Total=1.0). Target Risk=0.95.
    # Scores: A and B are low, C and D are topk.
    scores = pd.Series({"C": 0.9, "D": 0.8, "A": 0.1, "B": 0.1})
    current_pos = MockPosition({"A": 0.8, "B": 0.2})
    
    topk = 2
    risk_degree = 0.95
    impact_limit = 0.1 # Max change per step
    
    def create_test_strategy(priority):
        strat = SoftTopkStrategy.__new__(SoftTopkStrategy)
        strat.topk = topk
        strat.risk_degree = risk_degree
        strat.trade_impact_limit = impact_limit
        strat.priority = priority.upper()
        return strat

    # 1. Test IMPACT_FIRST: Expect deterministic sell and limited buy
    strat_i = create_test_strategy("IMPACT_FIRST")
    res_i = strat_i.generate_target_weight_position(scores, current_pos,None,None)
    
    # A should be exactly 0.8 - 0.1 = 0.7
    assert abs(res_i["A"] - 0.7) < 1e-8
    # B should be exactly 0.2 - 0.1 = 0.1
    assert abs(res_i["B"] - 0.1) < 1e-8
    # Total sells = 0.2 released. New budget = 0.2 + (0.95 - 1.0) = 0.15.
    # C and D share 0.15 -> 0.075 each.
    assert abs(res_i["C"] - 0.075) < 1e-8
    assert abs(res_i["D"] - 0.075) < 1e-8

    # 2. Test COMPLIANCE_FIRST: Expect full liquidation and full target fill
    strat_c = create_test_strategy("COMPLIANCE_FIRST")
    res_c = strat_c.generate_target_weight_position(scores, current_pos,None,None)
    
    # A, B not in topk -> Liquidated
    assert "A" not in res_c and "B" not in res_c
    # C, D should reach ideal_per_stock (0.95/2 = 0.475)
    assert abs(res_c["C"] - 0.475) < 1e-8
    assert abs(res_c["D"] - 0.475) < 1e-8

if __name__ == "__main__":
    pytest.main([__file__])