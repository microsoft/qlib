# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np

from ....model.meta.task import MetaTask
from ....data.dataset.handler import DataHandlerLP

from .utils import fill_diagnal, convert_data_to_tensor


class MetaTaskDS(MetaTask):
    """
    The MetaTask for the meta-learning-based data selection.
    """

    def __init__(self, task_def: dict, time_perf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_def = task_def
        self.time_perf = time_perf
        self._prepare_meta_task()

    def _prepare_meta_task(self):
        self.X, self.X_test = self.dataset.prepare(["train", "test"], col_set="feature", data_key=DataHandlerLP.DK_L)
        self.y, self.y_test = self.dataset.prepare(["train", "test"], col_set="label", data_key=DataHandlerLP.DK_L)
        self.sample_time_belong = np.zeros((self.y.shape[0], self.time_perf.shape[1]))
        for i, col in enumerate(self.time_perf.columns):
            slc = slice(*self.y.index.slice_locs(start=col[0], end=col[1]))
            self.sample_time_belong[slc, i] = 1.0
        # The last month also belongs to the last time_perf
        self.sample_time_belong[self.sample_time_belong.sum(axis=1) != 1, -1] = 1.0
        self.test_idx = self.y_test.index
        self.train_idx = self.y.index
        self.X, self.y, self.time_perf, self.sample_time_belong, self.X_test, self.y_test = convert_data_to_tensor(
            [self.X, self.y, self.time_perf, self.sample_time_belong, self.X_test, self.y_test]
        )

    def prepare_task_data(self):
        return (
            self.X,
            self.y,
            self.time_perf,
            self.sample_time_belong,
            self.X_test,
            self.y_test,
            self.test_idx,
            self.train_idx,
            self.task_def["outsample"],
        )
