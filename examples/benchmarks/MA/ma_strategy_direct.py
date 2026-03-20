# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Direct implementation of MA strategy using QLib API
"""
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.tests.data import GetData
import pandas as pd

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # Define MA strategy using QLib's built-in components
    from qlib.data import D
    from qlib.backtest import BacktestController, BacktestConfig
    from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
    import copy

    class MAStrategy(BaseSignalStrategy):
        def __init__(self, fast_period=5, slow_period=20, **kwargs):
            super().__init__(**kwargs)
            self.fast_period = fast_period
            self.slow_period = slow_period

        def generate_trade_decision(self, execute_result=None):
            # Get trading time
            trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
            
            # Get instruments
            instruments = self.trade_exchange.get_instrument_list()
            
            # Calculate MA signals
            # Get close prices
            close = D.features(instruments, ["$close"], start_time=trade_start_time, end_time=trade_end_time)
            if close.empty:
                return TradeDecisionWO([], self)
            
            # Calculate moving averages
            # For simplicity, we'll use a different approach since we can't easily calculate rolling means here
            # Instead, we'll use the built-in expression engine
            
            # Get MA5 and MA20
            ma5 = D.features(instruments, [f"Mean($close, {self.fast_period})"], start_time=trade_start_time, end_time=trade_end_time)
            ma20 = D.features(instruments, [f"Mean($close, {self.slow_period})"], start_time=trade_start_time, end_time=trade_end_time)
            
            if ma5.empty or ma20.empty:
                return TradeDecisionWO([], self)
            
            # Calculate signals
            ma5 = ma5.xs(trade_end_time, level="datetime")[f"Mean($close, {self.fast_period})"]
            ma20 = ma20.xs(trade_end_time, level="datetime")[f"Mean($close, {self.slow_period})"]
            
            # Generate signals: 1 for buy, -1 for sell
            signals = pd.Series(1, index=ma5.index)
            signals[ma5 < ma20] = -1
            
            # Get current position
            current_temp = copy.deepcopy(self.trade_position)
            sell_order_list = []
            buy_order_list = []
            
            # Get current holdings
            current_stock_list = current_temp.get_stock_list()
            cash = current_temp.get_cash()
            
            # Sell stocks with negative signals
            for code in current_stock_list:
                if code in signals.index and signals[code] < 0:
                    # Check if stock is tradable
                    if not self.trade_exchange.is_stock_tradable(
                        stock_id=code,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,
                    ):
                        continue
                    
                    # Create sell order
                    sell_amount = current_temp.get_stock_amount(code=code)
                    sell_order = Order(
                        stock_id=code,
                        amount=sell_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.SELL,
                    )
                    
                    # Check if order is executable
                    if self.trade_exchange.check_order(sell_order):
                        sell_order_list.append(sell_order)
                        trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                            sell_order, position=current_temp
                        )
                        cash += trade_val - trade_cost
            
            # Buy stocks with positive signals
            buy_stocks = signals[signals > 0].index.tolist()
            value = cash * self.risk_degree / len(buy_stocks) if len(buy_stocks) > 0 else 0
            
            for code in buy_stocks:
                # Check if stock is tradable
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                ):
                    continue
                
                # Create buy order
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
                )
                buy_amount = value / buy_price
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
                
                buy_order = Order(
                    stock_id=code,
                    amount=buy_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.BUY,
                )
                
                buy_order_list.append(buy_order)
            
            return TradeDecisionWO(sell_order_list + buy_order_list, self)

    # Create backtest config
    backtest_config = {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5
        }
    }

    # Create strategy
    strategy_config = {
        "class": "MAStrategy",
        "kwargs": {
            "fast_period": 5,
            "slow_period": 20,
            "risk_degree": 0.95
        }
    }

    # Create strategy instance
    from qlib.backtest.exchange import Exchange
    from qlib.backtest.calendar import CalendarManager
    
    # Create exchange
    exchange_config = {
        "class": "Exchange",
        "module_path": "qlib.backtest.exchange",
        "kwargs": backtest_config["exchange_kwargs"]
    }
    exchange = init_instance_by_config(exchange_config)
    
    # Create strategy
    strategy = MAStrategy(
        fast_period=5,
        slow_period=20,
        risk_degree=0.95,
        trade_exchange=exchange
    )

    # Create executor
    from qlib.backtest.executor import SimulatorExecutor
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        strategy=strategy,
        backtest=backtest_config
    )

    # Run backtest
    with R.start(experiment_name="MA_Strategy_Direct"):
        executor.run()
        
        # Get results
        port_analyzer = executor.get_portfolio_analyzer()
        print("Portfolio analysis results:")
        print(port_analyzer.get_analysis_result())
