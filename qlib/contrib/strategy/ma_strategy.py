# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd

from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class MAStrategy(BaseSignalStrategy):
    """Moving Average Strategy"""

    def __init__(
        self,
        *, 
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        risk_degree : float
            position percentage of total value.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
        """
        super().__init__(
            risk_degree=risk_degree,
            trade_exchange=trade_exchange,
            level_infra=level_infra,
            common_infra=common_infra,
            **kwargs,
        )

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        if pred_score is None:
            return TradeDecisionWO([], self)
        
        # NOTE: the current version of MA strategy uses the MA_SIGNAL column
        if isinstance(pred_score, pd.DataFrame):
            # 如果是DataFrame，使用MA_SIGNAL列
            if "MA_SIGNAL" in pred_score.columns:
                pred_score = pred_score["MA_SIGNAL"]
            else:
                # 否则使用第一列
                pred_score = pred_score.iloc[:, 0]
        
        current_temp = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        
        # 获取当前持仓
        current_stock_list = current_temp.get_stock_list()
        cash = current_temp.get_cash()
        
        # 基于信号生成交易决策
        # 买入信号: MA_SIGNAL = 1
        # 卖出信号: MA_SIGNAL = -1
        buy_stocks = pred_score[pred_score > 0].index.tolist()
        sell_stocks = pred_score[pred_score < 0].index.tolist()
        
        # 卖出当前持仓中需要卖出的股票
        for code in current_stock_list:
            if code in sell_stocks:
                # 检查股票是否可交易
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                ):
                    continue
                
                # 卖出订单
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,  # 0 for sell, 1 for buy
                )
                
                # 检查订单是否可执行
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    # 更新现金
                    cash += trade_val - trade_cost
        
        # 买入新股票
        # 计算每只股票的买入金额
        value = cash * self.risk_degree / len(buy_stocks) if len(buy_stocks) > 0 else 0
        
        for code in buy_stocks:
            # 检查股票是否可交易
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            ):
                continue
            
            # 买入订单
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
                direction=Order.BUY,  # 1 for buy
            )
            
            buy_order_list.append(buy_order)
        
        return TradeDecisionWO(sell_order_list + buy_order_list, self)
