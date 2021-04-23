# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import warnings
import numpy as np
import pandas as pd


from ..utils import get_sample_freq_calendar
from ..data.dataset import DatasetH
from ..backtest.order import Order
from ..backtest.env import TradeCalendarBase

"""
1. BaseStrategy 的粒度一定是数据粒度的整数倍
- 关于calendar的合并咋整
- adjust_dates这个东西啥用
- label和freq和strategy的bar分离，这个如何决策呢
"""
class BaseStrategy(TradeCalendarBase):
    def __init__(self, step_bar, start_time=None, end_time=None, **kwargs):
        self.step_bar = step_bar
        self.reset(start_time=start_time, end_time=end_time, **kwargs)

    def reset(self, start_time=None, end_time=None, _calendar=None, **kwargs):
        if start_time or end_time :
            self._reset_trade_calendar(start_time=start_time, end_time=end_time, calendar=calendar)
            
        for k, v in kwargs:
            if hasattr(self, k):
                setattr(self, k, v)
        
    
    def _get_trade_time(self):
        if 0 < self.trade_index < self.trade_len - 1: 
            trade_start_time = self.trade_calendar[self.trade_index - 1]
            trade_end_time = self.trade_calendar[self.trade_index] - pd.Timestamp(second=1)
            return trade_start_time, trade_end_time
        elif self.trade_index == self.trade_len - 1:
            trade_start_time = self.trade_calendar[self.trade_index - 1]
            trade_end_time = self.trade_calendar[self.trade_index]
            return trade_start_time, trade_end_time
        else:
            raise RuntimeError("trade_index out of range")
    
    def generate_order_list(self, **kwargs):
        self.trade_index = self.trade_index + 1


class RuleStrategy(BaseStrategy):
    pass

class DLStrategy(BaseStrategy):
    def __init__(self, step_bar, model, dataset:DatasetH, start_time=None, end_time=None):
        self.model = model
        self.dataset = dataset
        self.pred_scores = self.model.predict(dataset)
        #pred_score_dates = self.pred_scores.index.get_level_values(level="datetime")
        super(DLStrategy, self).__init__(step_bar, start_time, end_time)

     def _update_model(self):
        """update pred score
        """
        pass

class TradingEnhancement:
    def reset(self, trade_order_list):
        self.trade_order_list = trade_order_list

