# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pandas as pd
import numpy as np
from qlib.contrib.report.data.base import FeaAnalyser
from qlib.contrib.report.utils import sub_fig_generator
from qlib.utils.paral import datetime_groupby_apply
from qlib.contrib.eva.alpha import pred_autocorr_all
from loguru import logger
import seaborn as sns

DT_COL_NAME = "datetime"


class CombFeaAna(FeaAnalyser):
    """
    Combine the sub feature analysers and plot then in a single graph
    """

    def __init__(self, dataset: pd.DataFrame, *fea_ana_cls):
        if len(fea_ana_cls) <= 1:
            raise NotImplementedError(f"This type of input is not supported")
        self._fea_ana_l = [fcls(dataset) for fcls in fea_ana_cls]
        super().__init__(dataset=dataset)

    def skip(self, col):
        return np.all(list(map(lambda fa: fa.skip(col), self._fea_ana_l)))

    def calc_stat_values(self):
        """The statistics of features are finished in the underlying analysers"""

    def plot_all(self, *args, **kwargs):

        ax_gen = iter(sub_fig_generator(row_n=len(self._fea_ana_l), *args, **kwargs))

        for col in self._dataset:
            if not self.skip(col):
                axes = next(ax_gen)
                for fa, ax in zip(self._fea_ana_l, axes):
                    if not fa.skip(col):
                        fa.plot_single(col, ax)
                    ax.set_xlabel("")
                    ax.set_title("")
                axes[0].set_title(col)


class NumFeaAnalyser(FeaAnalyser):
    def skip(self, col):
        is_obj = np.issubdtype(self._dataset[col], np.dtype("O"))
        if is_obj:
            logger.info(f"{col} is not numeric and is skipped")
        return is_obj


class ValueCNT(FeaAnalyser):
    def __init__(self, dataset: pd.DataFrame, ratio=False):
        self.ratio = ratio
        super().__init__(dataset)

    def calc_stat_values(self):
        self._val_cnt = {}
        for col, item in self._dataset.items():
            if not super().skip(col):
                self._val_cnt[col] = item.groupby(DT_COL_NAME).apply(lambda s: len(s.unique()))
        self._val_cnt = pd.DataFrame(self._val_cnt)
        if self.ratio:
            self._val_cnt = self._val_cnt.div(self._dataset.groupby(DT_COL_NAME).size(), axis=0)

        # TODO: transfer this feature to other analysers
        ymin, ymax = self._val_cnt.min().min(), self._val_cnt.max().max()
        self.ylim = (ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))

    def plot_single(self, col, ax):
        self._val_cnt[col].plot(ax=ax, title=col, ylim=self.ylim)
        ax.set_xlabel("")


class FeaDistAna(NumFeaAnalyser):
    def plot_single(self, col, ax):
        sns.histplot(self._dataset[col], ax=ax, kde=False, bins=100)
        ax.set_xlabel("")
        ax.set_title(col)


class FeaInfAna(NumFeaAnalyser):
    def calc_stat_values(self):
        self._inf_cnt = {}
        for col, item in self._dataset.items():
            if not super().skip(col):
                self._inf_cnt[col] = item.apply(np.isinf).astype(np.int).groupby(DT_COL_NAME).sum()
        self._inf_cnt = pd.DataFrame(self._inf_cnt)

    def skip(self, col):
        return (col not in self._inf_cnt) or (self._inf_cnt[col].sum() == 0)

    def plot_single(self, col, ax):
        self._inf_cnt[col].plot(ax=ax, title=col)
        ax.set_xlabel("")


class FeaNanAna(FeaAnalyser):
    def calc_stat_values(self):
        self._nan_cnt = self._dataset.isna().groupby(DT_COL_NAME).sum()

    def skip(self, col):
        return (col not in self._nan_cnt) or (self._nan_cnt[col].sum() == 0)

    def plot_single(self, col, ax):
        self._nan_cnt[col].plot(ax=ax, title=col)
        ax.set_xlabel("")


class FeaNanAnaRatio(FeaAnalyser):
    def calc_stat_values(self):
        self._nan_cnt = self._dataset.isna().groupby(DT_COL_NAME).sum()
        self._total_cnt = self._dataset.groupby(DT_COL_NAME).size()

    def skip(self, col):
        return (col not in self._nan_cnt) or (self._nan_cnt[col].sum() == 0)

    def plot_single(self, col, ax):
        (self._nan_cnt[col] / self._total_cnt).plot(ax=ax, title=col)
        ax.set_xlabel("")


class FeaACAna(FeaAnalyser):
    """Analysis the auto-correlation of features"""

    def calc_stat_values(self):
        self._fea_corr = pred_autocorr_all(self._dataset.to_dict("series"))
        df = pd.DataFrame(self._fea_corr)
        ymin, ymax = df.min().min(), df.max().max()
        self.ylim = (ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))

    def plot_single(self, col, ax):
        self._fea_corr[col].plot(ax=ax, title=col, ylim=self.ylim)
        ax.set_xlabel("")


class FeaSkewTurt(NumFeaAnalyser):
    def calc_stat_values(self):
        self._skew = datetime_groupby_apply(self._dataset, "skew")
        self._kurt = datetime_groupby_apply(self._dataset, pd.DataFrame.kurt)

    def plot_single(self, col, ax):
        self._skew[col].plot(ax=ax, label="skew")
        ax.set_xlabel("")
        ax.set_ylabel("skew")
        ax.legend()

        right_ax = ax.twinx()

        self._kurt[col].plot(ax=right_ax, label="kurt", color="green")
        right_ax.set_xlabel("")
        right_ax.set_ylabel("kurt")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = right_ax.get_legend_handles_labels()

        ax.legend().set_visible(False)
        right_ax.legend(h1 + h2, l1 + l2)
        ax.set_title(col)


class FeaMeanStd(NumFeaAnalyser):
    def calc_stat_values(self):
        self._std = self._dataset.groupby(DT_COL_NAME).std()
        self._mean = self._dataset.groupby(DT_COL_NAME).mean()

    def plot_single(self, col, ax):
        self._mean[col].plot(ax=ax, label="mean")
        ax.set_xlabel("")
        ax.set_ylabel("mean")
        ax.legend()

        right_ax = ax.twinx()

        self._std[col].plot(ax=right_ax, label="std", color="green")
        right_ax.set_xlabel("")
        right_ax.set_ylabel("std")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = right_ax.get_legend_handles_labels()

        ax.legend().set_visible(False)
        right_ax.legend(h1 + h2, l1 + l2)
        ax.set_title(col)


class RawFeaAna(FeaAnalyser):
    """
    Motivation:
    - display the values without further analysis
    """

    def calc_stat_values(self):
        ymin, ymax = self._dataset.min().min(), self._dataset.max().max()
        self.ylim = (ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))

    def plot_single(self, col, ax):
        self._dataset[col].plot(ax=ax, title=col, ylim=self.ylim)
        ax.set_xlabel("")
