# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
import warnings
import numpy as np
import pandas as pd


from ..utils import get_sample_freq_calendar
from ..data.dataset import DatasetH
from ..data.dataset.utils import get_level_index
from ..contrib.backtest.order import Order
from ..contrib.backtest.env import BaseTradeCalendar

"""
1. BaseStrategy 的粒度一定是数据粒度的整数倍
- 关于calendar的合并咋整
- adjust_dates这个东西啥用
- label和freq和strategy的bar分离，这个如何决策呢
"""


class BaseStrategy(BaseTradeCalendar):
    def generate_order_list(self, **kwargs):
        raise NotImplementedError("generator_order_list is not implemented!")


class RuleStrategy(BaseStrategy):
    pass


class ModelStrategy(BaseStrategy):
    def __init__(self, step_bar, model, dataset: DatasetH, start_time=None, end_time=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.pred_scores = self._convert_index_format(self.model.predict(dataset))
        # pred_score_dates = self.pred_scores.index.get_level_values(level="datetime")
        super(ModelStrategy, self).__init__(step_bar, start_time, end_time, **kwargs)

    def _convert_index_format(self, df):
        if get_level_index(df, level="datetime") == 1:
            df = df.swaplevel().sort_index()
        return df

    def _update_model(self):
        """update pred score"""
        raise NotImplementedError("_update_model is not implemented!")


class TradingEnhancement:
    def reset(self, trade_order_list=None):
        if trade_order_list:
            self.trade_order_list = trade_order_list
