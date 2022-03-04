# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module is responsible for analysing data

Assumptions
- The analyse each feature individually

"""
import pandas as pd
from qlib.log import TimeInspector
from qlib.contrib.report.utils import sub_fig_generator


class FeaAnalyser:
    def __init__(self, dataset: pd.DataFrame):
        self._dataset = dataset
        with TimeInspector.logt("calc_stat_values"):
            self.calc_stat_values()

    def calc_stat_values(self):
        pass

    def plot_single(self, col, ax):
        raise NotImplementedError(f"This type of input is not supported")

    def skip(self, col):
        return False

    def plot_all(self, *args, **kwargs):

        ax_gen = iter(sub_fig_generator(*args, **kwargs))
        for col in self._dataset:
            if not self.skip(col):
                ax = next(ax_gen)
                self.plot_single(col, ax)
