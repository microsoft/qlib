# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Final implementation of MA strategy using QLib API
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

    # Create a simple MA strategy using built-in components
    from qlib.data import D
    from qlib.backtest import backtest
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
    from qlib.backtest.exchange import Exchange
    import copy

    class MAStrategy(BaseStrategy):
        def __init__(self, fast_period=5, slow_period=20, risk_degree=0.95, **kwargs):
            super().__init__(**kwargs)
            self.fast_period = fast_period
            self.slow_period = slow_period
            self.risk_degree = risk_degree

        def generate_trade_decision(self, execute_result=None):
            # Get trading time
            trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
            
            # Get instruments from QLib's data API
            from qlib.data import D
            instruments = D.instruments("csi300")
            
            if not instruments:
                return TradeDecisionWO([], self)
            
            # Calculate MA signals using QLib's expression engine
            try:
                # Calculate MA5 and MA20
                ma5_expr = f"Mean($close, {self.fast_period})"
                ma20_expr = f"Mean($close, {self.slow_period})"
                
                # Get moving averages
                ma5 = D.features(instruments, [ma5_expr], start_time=trade_start_time, end_time=trade_end_time)
                ma20 = D.features(instruments, [ma20_expr], start_time=trade_start_time, end_time=trade_end_time)
                
                if ma5.empty or ma20.empty:
                    return TradeDecisionWO([], self)
                
                # Extract values for the current trading day
                ma5_values = ma5.xs(trade_end_time, level="datetime")[ma5_expr]
                ma20_values = ma20.xs(trade_end_time, level="datetime")[ma20_expr]
                
                # Generate signals: 1 for buy, -1 for sell
                signals = pd.Series(1, index=ma5_values.index)
                signals[ma5_values < ma20_values] = -1
                
            except Exception as e:
                print(f"Error calculating MA signals: {e}")
                return TradeDecisionWO([], self)
            
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
                try:
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
                except Exception as e:
                    print(f"Error creating buy order for {code}: {e}")
                    continue
            
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

    # Run backtest using QLib's backtest function
    from qlib.backtest.backtest import backtest_loop
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.exchange import Exchange

    # Create exchange
    exchange = Exchange(**backtest_config["exchange_kwargs"])

    # Create strategy
    strategy = MAStrategy(
        fast_period=5,
        slow_period=20,
        risk_degree=0.95,
        trade_exchange=exchange
    )

    # Create executor
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        strategy=strategy,
        backtest=backtest_config
    )

    # Run backtest
    print("Running MA strategy backtest...")
    portfolio_dict, indicator_dict = backtest_loop(
        start_time=backtest_config["start_time"],
        end_time=backtest_config["end_time"],
        trade_strategy=strategy,
        trade_executor=executor
    )
    
    # Get results
    print("Backtest completed. Getting results...")
    print("Portfolio metrics:")
    for key, value in portfolio_dict.items():
        print(f"{key}:")
        print(value[0])
    print("\nIndicator analysis:")
    for key, value in indicator_dict.items():
        print(f"{key}:")
        print(value[0])
