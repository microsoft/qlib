# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import time
import copy
from typing import Union, List, Tuple, Text

from ....data.dataset import DatasetH
from ....data.dataset.handler import DataHandlerLP
from ....data import D
from ....utils import init_instance_by_config
from ....workflow.task.utils import TimeAdjuster
from ....model.meta.dataset import MetaDatasetH

from .utils import fill_diagnal
from .task import MetaTaskDS


class MetaDatasetHDS(MetaDatasetH):
    """
    The MetaDatasetH for the meta-Learning-based data selection.
    """

    def __init__(self, rolling_dict: dict, sim_mat=None, rolling_len=20, horizon=20, HIST_N=30, *args, **kwargs):
        """

        Parameters
        ----------
        rolling_dict: dict
            A dict that defines the train, valid (training for meta-model), and test scope.
        sim_mat: Union[pd.Dataframe, NoneType]
            The similarity matrix. The similarity matrix will be calculated if None is passed in.
        rolling_len: int
            The length of the test period in each rolling task.
        horizon: int
            The horizon of the label, the rolling process will create a gap between the training data and test data in order to avoid accessing the future information.
        HIST_N: int
            The number of periods that the meta-model will use.
        """
        super().__init__(*args, **kwargs)
        self.rolling_len = rolling_len
        self.rolling_dict = rolling_dict
        self.horizon = horizon
        self.HIST_N = HIST_N
        if sim_mat is None:
            self._init_sim_mat()
        else:
            self.sim_mat = sim_mat
        self.meta_tasks_l = self._generate_tasks_from_sim_mat()
        self.meta_tasks = self._init_meta_task_list()

    def _generate_tasks_from_sim_mat(self):
        ta = TimeAdjuster()
        cal = ta.cals
        dates = list(cal)
        meta_tasks_l = []
        rolling_start = self.rolling_dict["dataset"]["kwargs"]["segments"]["valid"][0]
        for (start, end) in self.sim_mat.columns:
            if start >= pd.Timestamp(rolling_start):
                meta_task = copy.deepcopy(self.rolling_dict)["dataset"]  # Be careful!
                rolling_start_idx = ta.align_idx(start)
                train_end = ta.get(rolling_start_idx - self.horizon)
                meta_task["kwargs"]["segments"]["train"] = (
                    pd.Timestamp(meta_task["kwargs"]["segments"]["train"][0]),
                    train_end,
                )
                meta_task["kwargs"]["segments"]["test"] = (start, end)
                meta_task["kwargs"]["segments"].pop("valid")
                meta_tasks_l.append(meta_task)
        return meta_tasks_l

    def get_sim_mat_from_tasks(self):
        """
        Get the similarity matrix from the initialized tasks.
        """
        sim_mat = {}
        for task in self.sim_tasks:
            sim_mean_series = pd.Series(task["sim_mean"])
            sim_mat[task["train_period"]] = sim_mean_series
        sim_mat_df = pd.DataFrame(sim_mat)
        return sim_mat_df

    def _init_sim_mat(self):
        """
        Initialize the similarity matrix.
        """
        self._generate_sim_task()
        self._calc_sim_mat()
        self.sim_mat = self.get_sim_mat_from_tasks()

    def _generate_sim_task(self):
        """
        Generate the the definition of the similarity matrix.
        """
        ta = TimeAdjuster()
        cal = ta.cals
        dates = list(cal)
        self.sim_tasks = []
        rolling_dict = copy.deepcopy(self.rolling_dict)
        train_start, train_end = rolling_dict["dataset"]["kwargs"]["segments"]["train"]
        valid_start, valid_end = rolling_dict["dataset"]["kwargs"]["segments"]["valid"]
        test_start, test_end = rolling_dict["dataset"]["kwargs"]["segments"]["test"]
        train_start_idx, train_end_idx = ta.align_idx(train_start), ta.align_idx(train_end)
        valid_start_idx, valid_end_idx = ta.align_idx(valid_start), ta.align_idx(valid_end)
        test_start_idx, test_end_idx = ta.align_idx(test_start), ta.align_idx(test_end)
        start_idx = train_start_idx + ((test_start_idx - train_start_idx) % self.rolling_len)  # To align at test start

        def get_rolling_periods():
            rolling_periods = []
            if start_idx - 1 > train_start_idx:
                rolling_periods.append((dates[train_start_idx], dates[start_idx - 1]))
            for t_start, t_end in zip(
                dates[start_idx : test_end_idx + 1 : self.rolling_len],
                dates[start_idx + self.rolling_len - 1 : test_end_idx + 1 : self.rolling_len],
            ):
                rolling_periods.append((t_start, t_end))
            t_end_idx = ta.align_idx(t_end)
            if t_end_idx + 1 < test_end_idx:
                rolling_periods.append((dates[t_end_idx + 1], dates[test_end_idx]))
            return rolling_periods

        rolling_periods = get_rolling_periods()
        for period in rolling_periods:
            sim_task = {"train_period": period, "rolling_periods": rolling_periods}
            self.sim_tasks.append(sim_task)

    def _calc_sim_mat(self):
        """
        Calculate the similarity matrix.
        """
        print("Calculating the similarity matrix...")
        start_time = time.time()
        for index, task in enumerate(self.sim_tasks):
            # Prepare the dataset
            rolling_dict = copy.deepcopy(self.rolling_dict)
            task["dataset"] = rolling_dict["dataset"]
            task["dataset"]["kwargs"]["handler"] = self.data_handler
            task_seg = {
                "train": task["train_period"],
                "test": (task["rolling_periods"][0][0], task["rolling_periods"][-1][1]),
            }
            task["dataset"]["kwargs"]["segments"] = task_seg
            task["dataset"] = init_instance_by_config(task["dataset"])

            # Train & inference the model
            task["model"] = init_instance_by_config(rolling_dict["model"])
            task["model"].fit(task["dataset"])
            pred = task["model"].predict(task["dataset"])
            label = task["dataset"].prepare("test", col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0]

            # Calculate the similarity
            sim_mean = {}
            for (rolling_start, rolling_end) in task["rolling_periods"]:
                df = pd.DataFrame(
                    {"pred": pred.loc[rolling_start:rolling_end], "label": label.loc[rolling_start:rolling_end]}
                )
                sims = df.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
                sim_mean[(rolling_start, rolling_end)] = sims.mean()
            task["sim_mean"] = sim_mean
        end_time = time.time()
        print("The similarity matrix calculating process is finished. Total time: %.2fs." % (end_time - start_time))

    def _init_meta_task_list(self, *args, **kwargs):
        meta_tasks = []
        for task in self.meta_tasks_l:
            meta_task = self._init_meta_task(task)
            if meta_task is not None:
                meta_tasks.append(meta_task)
        if meta_tasks == []:
            raise AssertionError("No meta-task is created!")
        return meta_tasks

    def _init_meta_task(self, meta_task: dict, *args, **kwargs) -> MetaTaskDS:
        meta_task["kwargs"]["handler"] = self.data_handler
        test_date = meta_task["kwargs"]["segments"]["test"]
        sim_mat_fill = fill_diagnal(self.sim_mat)  # Remove the future information
        sim_mat_focus = sim_mat_fill.loc[:test_date, :test_date]

        task_def = {
            # Because the last month may leak future information, so -1 is excluded
            "insample": list(sim_mat_focus.index[:-2]),
            "outsample": test_date,  # sim_mat_focus.index[-1],
        }

        time_perf = None  # For possible spatical extension
        task_idx = len(sim_mat_focus)
        if task_idx > self.HIST_N:
            time_perf = sim_mat_focus.iloc[-self.HIST_N - 1 : -1].loc[:, task_def["insample"]]
        if time_perf is None:  # Only qualified meta-task will be created
            return None
        return MetaTaskDS(task_def, time_perf, meta_task)

    def _prepare_seg(self, segment: str, *args, **kwargs):
        assert len(self.meta_tasks_l) == len(self.meta_tasks)
        meta_tasks = []
        test_start_date = pd.Timestamp(self.rolling_dict["dataset"]["kwargs"]["segments"]["test"][0])
        for index, task_def in enumerate(self.meta_tasks_l):
            task_date = pd.Timestamp(task_def["kwargs"]["segments"]["test"][0])
            if (segment == "train" and task_date < test_start_date) or (
                segment == "test" and task_date >= test_start_date
            ):
                meta_tasks.append(self.meta_tasks[index])
        return meta_tasks

    def get_test_period_from_meta_tasks(self):
        return [task["kwargs"]["segments"]["test"] for task in self.meta_tasks_l]

    def get_meta_task_by_test_period(self, test_period: Union[list, tuple]):
        """
        Get the meta-task by the given key (test period). Return None if the meta-task is not found.
        Assume the task instances in meta_tasks and the task definitions in meta_tasks_l are corresponding.
        """
        # Find the exact one
        period_tuple = tuple([pd.Timestamp(t) for t in test_period])
        periods = self.get_test_period_from_meta_tasks()
        for index, key in enumerate(periods):
            if key == period_tuple:
                return self.meta_tasks[index]
        # If there is no exact one, find the nearest one
        nearest_idx = None
        for index, key in enumerate(periods):
            if key[0] <= period_tuple[0]:
                if nearest_idx is None or periods[nearest_idx][0] < key[0]:
                    nearest_idx = index
        if nearest_idx is not None:
            return self.meta_tasks[nearest_idx]
        return None
