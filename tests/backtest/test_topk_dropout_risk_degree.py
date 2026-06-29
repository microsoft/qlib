import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd


class _DummyPosition:
    def __init__(self, cash=100.0):
        self._cash = cash

    def get_cash(self):
        return self._cash

    def get_stock_list(self):
        return []

    def __deepcopy__(self, memo):
        return _DummyPosition(self._cash)


class _DummyTradeExchange:
    def is_stock_tradable(self, **kwargs):
        return True

    def get_deal_price(self, **kwargs):
        return 10.0

    def get_factor(self, **kwargs):
        return 1.0

    def round_amount_by_trade_unit(self, amount, factor):
        return amount


class TopkDropoutRiskDegreeTest(unittest.TestCase):
    @staticmethod
    def _load_topk_dropout_strategy():
        strategy_path = Path(__file__).resolve().parents[2] / "qlib" / "contrib" / "strategy" / "signal_strategy.py"

        qlib_pkg = ModuleType("qlib")
        qlib_pkg.__path__ = []
        contrib_pkg = ModuleType("qlib.contrib")
        contrib_pkg.__path__ = []
        contrib_strategy_pkg = ModuleType("qlib.contrib.strategy")
        contrib_strategy_pkg.__path__ = []
        backtest_pkg = ModuleType("qlib.backtest")
        backtest_pkg.__path__ = []
        data_pkg = ModuleType("qlib.data")
        data_pkg.__path__ = []
        dataset_pkg = ModuleType("qlib.data.dataset")
        model_pkg = ModuleType("qlib.model")
        model_pkg.__path__ = []
        model_base_pkg = ModuleType("qlib.model.base")
        strategy_pkg = ModuleType("qlib.strategy")
        strategy_pkg.__path__ = []
        strategy_base_pkg = ModuleType("qlib.strategy.base")
        backtest_position_pkg = ModuleType("qlib.backtest.position")
        backtest_signal_pkg = ModuleType("qlib.backtest.signal")
        backtest_decision_pkg = ModuleType("qlib.backtest.decision")
        log_pkg = ModuleType("qlib.log")
        utils_pkg = ModuleType("qlib.utils")
        order_generator_pkg = ModuleType("qlib.contrib.strategy.order_generator")
        optimizer_pkg = ModuleType("qlib.contrib.strategy.optimizer")

        data_pkg.D = object()
        dataset_pkg.Dataset = type("Dataset", (), {})
        model_base_pkg.BaseModel = type("BaseModel", (), {})

        class BaseStrategy:
            def __init__(self, *args, **kwargs):
                pass

        strategy_base_pkg.BaseStrategy = BaseStrategy
        backtest_position_pkg.Position = _DummyPosition

        class Signal:
            pass

        backtest_signal_pkg.Signal = Signal
        backtest_signal_pkg.create_signal_from = lambda signal: signal

        class Order:
            BUY = 1
            SELL = 0

            def __init__(self, stock_id, amount, start_time, end_time, direction):
                self.stock_id = stock_id
                self.amount = amount
                self.start_time = start_time
                self.end_time = end_time
                self.direction = direction

        class OrderDir:
            BUY = 1
            SELL = 0

        class TradeDecisionWO:
            def __init__(self, order_list, strategy):
                self.order_list = order_list
                self.strategy = strategy

        backtest_decision_pkg.Order = Order
        backtest_decision_pkg.OrderDir = OrderDir
        backtest_decision_pkg.TradeDecisionWO = TradeDecisionWO

        log_pkg.get_module_logger = lambda name: SimpleNamespace(info=lambda *a, **k: None)
        utils_pkg.get_pre_trading_date = lambda *args, **kwargs: None
        utils_pkg.load_dataset = lambda *args, **kwargs: None
        order_generator_pkg.OrderGenerator = type("OrderGenerator", (), {})
        order_generator_pkg.OrderGenWOInteract = type("OrderGenWOInteract", (), {})
        optimizer_pkg.EnhancedIndexingOptimizer = type("EnhancedIndexingOptimizer", (), {})

        with patch.dict(
            "sys.modules",
            {
                "qlib": qlib_pkg,
                "qlib.contrib": contrib_pkg,
                "qlib.contrib.strategy": contrib_strategy_pkg,
                "qlib.backtest": backtest_pkg,
                "qlib.data": data_pkg,
                "qlib.data.dataset": dataset_pkg,
                "qlib.model": model_pkg,
                "qlib.model.base": model_base_pkg,
                "qlib.strategy": strategy_pkg,
                "qlib.strategy.base": strategy_base_pkg,
                "qlib.backtest.position": backtest_position_pkg,
                "qlib.backtest.signal": backtest_signal_pkg,
                "qlib.backtest.decision": backtest_decision_pkg,
                "qlib.log": log_pkg,
                "qlib.utils": utils_pkg,
                "qlib.contrib.strategy.order_generator": order_generator_pkg,
                "qlib.contrib.strategy.optimizer": optimizer_pkg,
            },
        ):
            spec = spec_from_file_location("qlib.contrib.strategy.signal_strategy", strategy_path)
            module = module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module.TopkDropoutStrategy

    def test_generate_trade_decision_uses_dynamic_risk_degree_for_buy_sizing(self):
        TopkDropoutStrategy = self._load_topk_dropout_strategy()

        strategy = TopkDropoutStrategy.__new__(TopkDropoutStrategy)
        strategy.topk = 1
        strategy.n_drop = 1
        strategy.method_sell = "bottom"
        strategy.method_buy = "top"
        strategy.hold_thresh = 1
        strategy.only_tradable = False
        strategy.forbid_all_trade_at_limit = True
        strategy.risk_degree = 0.95
        strategy.trade_position = _DummyPosition(cash=100.0)
        strategy.trade_exchange = _DummyTradeExchange()
        strategy.trade_calendar = SimpleNamespace(
            get_trade_step=lambda: 0,
            get_step_time=lambda trade_step=None, shift=0: (
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02 23:59:59"),
            ),
            get_freq=lambda: "day",
        )
        strategy.signal = SimpleNamespace(
            get_signal=lambda start_time=None, end_time=None: pd.Series({"A": 1.0}, name="score")
        )
        strategy.get_risk_degree = lambda trade_step=None: 0.0

        decision = strategy.generate_trade_decision()

        self.assertEqual(1, len(decision.order_list))
        self.assertEqual(0.0, decision.order_list[0].amount)


if __name__ == "__main__":
    unittest.main()
