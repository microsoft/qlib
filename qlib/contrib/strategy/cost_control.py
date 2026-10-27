# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
            Maximum weight change for each stock in one trade. If None, fallback to max_sold_weight.
        max_sold_weight : float
            Backward-compatible alias for trade_impact_limit. Use 1.0 to effectively disable the limit.
        risk_degree : float
            The target percentage of total value to be invested.
        """
        super(SoftTopkStrategy, self).__init__(
            model=model, dataset=dataset, order_generator_cls_or_obj=order_generator_cls_or_obj, **kwargs
        )

        self.topk = topk
        self.trade_impact_limit = trade_impact_limit if trade_impact_limit is not None else max_sold_weight
        self.risk_degree = risk_degree
        self.buy_method = buy_method

    def get_risk_degree(self, trade_step=None):
        return self.risk_degree

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time, **kwargs):
        """
        Generates target position using Proportional Budget Allocation.
        Ensures deterministic sells and synchronized buys under impact limits.

        ``get_risk_degree`` receives the current trade step when the calendar
        has started (step >= 0). It receives None when there is no calendar,
        and also when ``level_infra`` is set but the calendar has not started:
        a negative step, a non-int step, or ``get_trade_step()`` raising
        ``AttributeError`` because the step was never initialized.
        """

        if self.topk is None or self.topk <= 0:
            return {}

        def apply_impact_limit(weight):
            return weight if self.trade_impact_limit is None else min(weight, self.trade_impact_limit)

        # Use the getter so a subclass can vary exposure. Reading the attribute
        # here made an override of get_risk_degree() unreachable, while
        # WeightStrategyBase already passes that override to the order generator.
        # Pass the same trade step the order generator receives, otherwise an
        # override that depends on the step sees None here and a real index there.
        # A calendar that has not started stays on the None path. Forwarding a
        # negative step (or letting an uninitialized get_trade_step() raise) is
        # the one place a getter that indexes by step can raise where the old
        # no-argument call did not. Step 0 is the first bar and is forwarded.
        trade_step = None
        if getattr(self, "level_infra", None) is not None and self.trade_calendar is not None:
            try:
                candidate = self.trade_calendar.get_trade_step()
            except AttributeError:
                candidate = None
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                trade_step = candidate
        risk_degree = self.get_risk_degree(trade_step)
        ideal_per_stock = risk_degree / self.topk
        ideal_list = score.sort_values(ascending=False).iloc[: self.topk].index.tolist()

        cur_weights = current.get_stock_weight_dict(only_stock=True)
        initial_total_weight = sum(cur_weights.values())

        # --- Case A: Cold Start ---
        if not cur_weights:
            fill = apply_impact_limit(ideal_per_stock)
            return {code: fill for code in ideal_list}

        # --- Case B: Rebalancing ---
        all_tickers = set(cur_weights.keys()) | set(ideal_list)
        next_weights = {t: cur_weights.get(t, 0.0) for t in all_tickers}

        # Phase 1: Deterministic Sell Phase
        released_cash = 0.0
        for t in list(next_weights.keys()):
            cur = next_weights[t]
            if cur <= 1e-8:
                continue

            if t not in ideal_list:
                sell = apply_impact_limit(cur)
                next_weights[t] -= sell
                released_cash += sell
            elif cur > ideal_per_stock + 1e-8:
                excess = cur - ideal_per_stock
                sell = apply_impact_limit(excess)
                next_weights[t] -= sell
                released_cash += sell

        # Phase 2: Budget Calculation
        # Budget = Cash from sells + Available space from target risk degree
        total_budget = released_cash + (risk_degree - initial_total_weight)

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

                    # Capped by impact limit
                    max_buy_cap = apply_impact_limit(shortfall)

                    next_weights[t] += min(share_of_budget, max_buy_cap)

        return {k: v for k, v in next_weights.items() if v > 1e-8}
