# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import pandas as pd
from ...data import D
from ..model.base import Model


class ScoreFileModel(Model):
    """
    This model will load a score file, and return score at date exists in score file.
    """

    def __init__(self, score_path):
        pred_test = pd.read_csv(score_path, index_col=[0, 1], parse_dates=True, infer_datetime_format=True)
        self.pred = pred_test

    def get_data_with_date(self, date, **kwargs):
        score = self.pred.loc(axis=0)[:, date]  # (stock_id, trade_date) multi_index, score in pdate
        score_series = score.reset_index(level="datetime", drop=True)[
            "score"
        ]  # pd.Series ; index:stock_id, data: score
        return score_series

    def predict(self, x_test, **kwargs):
        return x_test

    def score(self, x_test, **kwargs):
        return

    def fit(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs):
        return

    def save(self, fname, **kwargs):
        return
