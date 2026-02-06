import numpy as np
from copy import deepcopy

from typing import Dict, List, Union, Text, Tuple

from qlib.data.dataset.utils import init_task_handler
from qlib.data.dataset import Dataset, TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.meta.dataset import MetaTaskDataset
from qlib.log import get_module_logger
from qlib.utils import init_instance_by_config
from qlib.utils.data import deepcopy_basic_type
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.utils import TimeAdjuster
from tqdm.auto import tqdm

from .utils import get_data_from_seg, get_data_and_idx


class MetaTaskInc:
    """Meta task for incremental learning"""

    def __init__(self, task: dict, data=None, data_I=None, mode="train"):
        self.task = task

        train_exist = "train" in self.task["dataset"]["kwargs"]["segments"]
        extra_exist = "extra" in self.task["dataset"]["kwargs"]["segments"]
        if train_exist:
            train_segs = [str(dt) for dt in self.task["dataset"]["kwargs"]["segments"]["train"]]
        if extra_exist:
            extra_segs = [str(dt) for dt in self.task["dataset"]["kwargs"]["segments"]["extra"]]
        test_segs = [str(dt) for dt in self.task["dataset"]["kwargs"]["segments"]["test"]]
        if isinstance(data, TSDataSampler):
            if train_exist:
                d_train, d_train_idx = get_data_and_idx(data, train_segs)
            if extra_exist:
                d_extra, d_extra_idx = get_data_and_idx(data, extra_segs)
            d_test, d_test_idx = get_data_and_idx(data, test_segs)
            self.processed_meta_input = dict(X_test=d_test[:, :, 0:-1], y_test=d_test[:, -1, -1], test_idx=d_test_idx,)
            if train_exist:
                self.processed_meta_input.update(
                    X_train=d_train[:, :, 0:-1], y_train=d_train[:, -1, -1], train_idx=d_train_idx,
                )
            if extra_exist:
                self.processed_meta_input.update(
                    X_extra=d_extra[:, :, 0:-1], y_extra=d_extra[:, -1, -1], extra_idx=d_extra_idx,
                )
        else:
            if data is None:
                ds = init_instance_by_config(self.task["dataset"], accept_types=Dataset)
                if train_exist:
                    d_train = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)
                if extra_exist:
                    d_extra = ds.prepare("extra", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)
                d_test = ds.prepare(
                    "test",
                    col_set=["feature", "label"],
                    data_key=DataHandlerLP.DK_L if mode == "train" else DataHandlerLP.DK_I,
                )
                if mode == "test":
                    assert (
                        np.isnan(d_test["feature"].values).sum() == 0
                    ), f'Check null test segment {self.task["dataset"]["kwargs"]["segments"]["test"]}'
            else:
                if train_exist:
                    d_train = get_data_from_seg(train_segs, data)
                if extra_exist:
                    d_extra = get_data_from_seg(extra_segs, data)
                if mode != "train" and data_I is not None:
                    data = data_I
                d_test = get_data_from_seg(test_segs, data, True)
            self.processed_meta_input = dict(
                X_test=d_test["feature"], y_test=d_test["label"].iloc[:, 0], test_idx=d_test["label"].index,
            )
            if train_exist:
                self.processed_meta_input.update(
                    X_train=d_train["feature"], y_train=d_train["label"].iloc[:, 0], train_idx=d_train["label"].index,
                )
            if extra_exist:
                self.processed_meta_input.update(
                    X_extra=d_extra["feature"], y_extra=d_extra["label"].iloc[:, 0], extra_idx=d_extra["label"].index,
                )

    def get_meta_input(self):
        return self.processed_meta_input


class MetaDatasetInc(MetaTaskDataset):
    def __init__(
        self,
        *,
        task_tpl: Union[dict, list],
        step: int,
        trunc_days: int = None,
        rolling_ext_days: int = 0,
        segments: Union[Dict[Text, Tuple], float],
        task_mode: str = "train",
        data=None,
        data_I=None,
    ):
        """
        A dataset for meta model.

        Parameters
        ----------
        task_tpl : Union[dict, list]
            Decide what tasks are used.
            - dict : the task template, the prepared task is generated with `step`, `trunc_days` and `RollingGen`
            - list : when list, use the list of tasks directly
                     the list is supposed to be sorted according timeline
        step : int
            the rolling step
        trunc_days: int
            days to be truncated based on the test start
        rolling_ext_days: int
            sometimes users want to train meta models for a longer test period but with smaller rolling steps for more task samples.
            the total length of test periods will be `step + rolling_ext_days`
        segments: Union[Dict[Text, Tuple], float]
            the segments to divide data
            both left and right
            if segments is a float:
                the float represents the percentage of data for training
        hist_step_n: int
            length of historical steps for the meta infomation
        task_mode : str
            If 'test', use data_I, especially for MLP on Alpha158 without dropping nan labels.
        """
        super().__init__(segments=segments)
        self.task_tpl = deepcopy_basic_type(task_tpl)
        self.trunc_days = trunc_days
        self.step = step

        if isinstance(task_tpl, dict):
            rg = RollingGen(
                step=step, trunc_days=trunc_days, task_copy_func=deepcopy_basic_type, rtype=RollingGen.ROLL_SD,
            )  # NOTE: trunc_days is very important !!!!
            task_iter = rg(task_tpl)
            if rolling_ext_days > 0:
                self.ta = TimeAdjuster(future=True)
                for t in task_iter:
                    t["dataset"]["kwargs"]["segments"]["test"] = self.ta.shift(
                        t["dataset"]["kwargs"]["segments"]["test"], step=rolling_ext_days, rtype=RollingGen.ROLL_EX,
                    )
            # init_task_handler(task_tpl)
        else:
            assert isinstance(task_tpl, list)
            task_iter = task_tpl

        self.task_list = []
        self.meta_task_l = []
        logger = get_module_logger("MetaDatasetInc")
        logger.info(f"Example task for training meta model: {task_iter[0]}")

        for t in tqdm(task_iter, desc="creating meta tasks"):
            self.meta_task_l.append(MetaTaskInc(t, data=data, data_I=data_I, mode=task_mode))
            self.task_list.append(t)
            assert len(self.meta_task_l) > 0, "No meta tasks found. Please check the data and setting"

    def _prepare_seg(self, segment: Text) -> List[MetaTaskInc]:
        if isinstance(self.segments, float):
            train_task_n = int(len(self.meta_task_l) * self.segments)
            if segment == "train":
                return self.meta_task_l[:train_task_n]
            elif segment == "test":
                return self.meta_task_l[train_task_n:]
            else:
                raise NotImplementedError(f"This type of input is not supported")
        elif isinstance(self.segments, str):
            for i, t in enumerate(self.meta_task_l):
                if t.task["dataset"]["kwargs"]["segments"]["test"][-1]._date_repr >= self.segments:
                    break
            if segment == "train":
                return self.meta_task_l[:i]
            elif segment == "test":
                return self.meta_task_l[i:]
            else:
                raise NotImplementedError(f"This type of input is not supported")
        else:
            raise NotImplementedError(f"This type of input is not supported")
