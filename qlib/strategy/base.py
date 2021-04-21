# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import warnings
import numpy as np
import pandas as pd


from ..utils import sample_feature, get_sample_freq_calendar
from ..data.dataset import DatasetH
from ..backtest.order import Order
from .order_generator import OrderGenWInteract
from ..data.data import D
"""
1. BaseStrategy 的粒度一定是数据粒度的整数倍
- 关于calendar的合并咋整
- adjust_dates这个东西啥用
- label和freq和strategy的bar分离，这个如何决策呢
"""
class BaseStrategy:
    def __init__(self, step_bar, start_time, end_time, **kwargs):
        self.step_bar = step_bar
        self.reset(start_time=start_time, end_time=end_time, **kwargs)

    def _reset_trade_date(self, start_time=None, end_time=None):
        if start_time:
            self.start_time = start_time
        if end_time:
            self.end_time = end_time
        if not self.start_time or not self.end_time:
            raise ValueError("value of `start_time` or `end_time` is None")
        _calendar = get_sample_freq_calendar(start_time=start_time, end_time=end_time, freq=step_bar)
        self.trade_dates = np.hstack(_calendar, pd.Timestamp(self.end_time))
        self.trade_len = len(self.trade_dates)
        self.trade_index = 0
        
    def reset(self, start_time=None, end_time=None, **kwargs):
        if start_time or end_time:
            self._reset_trade_date(start_time=start_time, end_time=end_time)

    def generate_order_list(self, **kwargs):
        self.trade_index = self.trade_index + 1


class RuleStrategy(BaseStrategy):
    pass

class DLStrategy(BaseStrategy):
    def __init__(self, step_bar, start_time, end_time, model, dataset:DatasetH):
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
        if trade_order_list:
            self.trade_order_list = trade_order_list

