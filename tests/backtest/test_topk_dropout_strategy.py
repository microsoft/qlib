import pandas as pd

from qlib.backtest.decision import Order
from qlib.backtest.position import Position
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


class MockInfra:
    def __init__(self, values):
        self.values = values

    def get(self, name):
        return self.values[name]


class MockAccount:
    def __init__(self, position):
        self.current_position = position


class MockCalendar:
    def get_trade_step(self):
        return 0

    def get_step_time(self, trade_step=None, shift=0):
        if shift:
            return pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")
        return pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02")


class MockSignal:
    def get_signal(self, start_time, end_time):
        return pd.Series({"A": 1.0})


class MockExchange:
    def is_stock_tradable(self, stock_id, start_time, end_time, direction=None):
        return True

    def get_deal_price(self, stock_id, start_time, end_time, direction):
        return 10.0

    def get_factor(self, stock_id, start_time, end_time):
        return 1.0

    def round_amount_by_trade_unit(self, amount, factor):
        return amount


class DynamicRiskTopkDropoutStrategy(TopkDropoutStrategy):
    def get_risk_degree(self, trade_step=None):
        return 0.25


def test_topk_dropout_uses_dynamic_risk_degree_for_buy_budget():
    strategy = DynamicRiskTopkDropoutStrategy.__new__(DynamicRiskTopkDropoutStrategy)
    strategy.signal = MockSignal()
    strategy.risk_degree = 0.95
    strategy.topk = 1
    strategy.n_drop = 1
    strategy.method_sell = "bottom"
    strategy.method_buy = "top"
    strategy.hold_thresh = 1
    strategy.only_tradable = False
    strategy.forbid_all_trade_at_limit = True
    strategy.level_infra = MockInfra({"trade_calendar": MockCalendar()})
    strategy.common_infra = MockInfra(
        {"trade_account": MockAccount(Position(cash=1000.0))}
    )
    strategy._trade_exchange = MockExchange()

    decision = strategy.generate_trade_decision()
    orders = decision.get_decision()

    assert len(orders) == 1
    assert orders[0].stock_id == "A"
    assert orders[0].direction == Order.BUY
    assert orders[0].amount == 25.0
