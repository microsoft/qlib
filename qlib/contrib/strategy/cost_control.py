# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from .order_generator import OrderGenWInteract
from .signal_strategy import WeightStrategyBase

class SoftTopkStrategy(WeightStrategyBase):
    def __init__(
        self,
        model=None,
        dataset=None,
        topk=None,
        order_generator_cls_or_obj=OrderGenWInteract,
        max_sold_weight=1.0,
        trade_impact_limit=None,
        priority="IMPACT_FIRST",
        risk_degree=0.95,
        buy_method="first_fill",
        **kwargs,
    ):
        """
        Refactored SoftTopkStrategy with a budget-constrained rebalancing engine.
        
        Parameters
        ----------
        topk : int
            The number of top-N stocks to be held in the portfolio.
        trade_impact_limit : float
            Maximum weight change for each stock in one trade. 
        priority : str
            "COMPLIANCE_FIRST" or "IMPACT_FIRST".
        risk_degree : float
            The target percentage of total value to be invested.
        """
        super(SoftTopkStrategy, self).__init__(
            model=model, 
            dataset=dataset, 
            order_generator_cls_or_obj=order_generator_cls_or_obj, 
            **kwargs
        )
        
        self.topk = topk
        self.trade_impact_limit = trade_impact_limit if trade_impact_limit is not None else max_sold_weight
        self.priority = priority.upper()
        self.risk_degree = risk_degree
        self.buy_method = buy_method

    def get_risk_degree(self, trade_step=None):
        return self.risk_degree

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time, **kwargs):
        """
        Generates target position using Proportional Budget Allocation.
        Ensures deterministic sells and synchronized buys under impact limits.
        """

        if self.topk is None or self.topk <= 0:
            return {}

        ideal_per_stock = self.risk_degree / self.topk
        ideal_list = score.sort_values(ascending=False).iloc[:self.topk].index.tolist()
        
        cur_weights = current.get_stock_weight_dict(only_stock=True)
        initial_total_weight = sum(cur_weights.values())
        
        # --- Case A: Cold Start ---
        if not cur_weights:
            fill = ideal_per_stock if self.priority == "COMPLIANCE_FIRST" else min(ideal_per_stock, self.trade_impact_limit)
            return {code: fill for code in ideal_list}

        # --- Case B: Rebalancing ---
        all_tickers = set(cur_weights.keys()) | set(ideal_list)
        next_weights = {t: cur_weights.get(t, 0.0) for t in all_tickers}
        
        # Phase 1: Deterministic Sell Phase
        released_cash = 0.0
        for t in list(next_weights.keys()):
            cur = next_weights[t]
            if cur <= 1e-8: continue
            
            if t not in ideal_list:
                sell = cur if self.priority == "COMPLIANCE_FIRST" else min(cur, self.trade_impact_limit)
                next_weights[t] -= sell
                released_cash += sell
            elif cur > ideal_per_stock + 1e-8:
                excess = cur - ideal_per_stock
                sell = excess if self.priority == "COMPLIANCE_FIRST" else min(excess, self.trade_impact_limit)
                next_weights[t] -= sell
                released_cash += sell

        # Phase 2: Budget Calculation
        # Budget = Cash from sells + Available space from target risk degree
        total_budget = released_cash + (self.risk_degree - initial_total_weight)
        
        # Phase 3: Proportional Buy Allocation
        if total_budget > 1e-8:
            shortfalls = {
                t: (ideal_per_stock - next_weights.get(t, 0.0))
                for t in ideal_list
                if next_weights.get(t, 0.0) < ideal_per_stock - 1e-8
            }
            
            if shortfalls:
                total_shortfall = sum(shortfalls.values())
                # Normalize total_budget to not exceed total_shortfall
                available_to_spend = min(total_budget, total_shortfall)
                
                for t, shortfall in shortfalls.items():
                    # Every stock gets its fair share based on its distance to target
                    share_of_budget = (shortfall / total_shortfall) * available_to_spend
                    
                    # Capped by impact limit or compliance priority
                    max_buy_cap = shortfall if self.priority == "COMPLIANCE_FIRST" else min(shortfall, self.trade_impact_limit)
                    
                    next_weights[t] += min(share_of_budget, max_buy_cap)

        return {k: v for k, v in next_weights.items() if v > 1e-8}