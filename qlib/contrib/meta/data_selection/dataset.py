# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pandas as pd
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed  # pylint: disable=E0401
from typing import Dict, List, Union, Text, Tuple
from qlib.data.dataset.utils import init_task_handler
from qlib.data.dataset import DatasetH
from qlib.contrib.torch import data_to_tensor
from qlib.model.meta.task import MetaTask
from qlib.model.meta.dataset import MetaTaskDataset
from qlib.model.trainer import TrainerR
from qlib.log import get_module_logger
from qlib.utils import auto_filter_kwargs, get_date_by_shift, init_instance_by_config
from qlib.utils.data import deepcopy_basic_type
from qlib.workflow import R
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.utils import TimeAdjuster
from tqdm.auto import tqdm


class InternalData:
    def __init__(self, task_tpl: dict, step: int, exp_name: str):
        self.task_tpl = task_tpl
        self.step = step
        self.exp_name = exp_name

    def setup(self, trainer=TrainerR, trainer_kwargs={}):
        """
        after running this function `self.data_ic_df` will become set.
        Each col represents a data.
        Each row represents the Timestamp of performance of that data.
        For example,

        .. code-block:: python

                       2021-06-21 2021-06-04 2021-05-21 2021-05-07 2021-04-20 2021-04-06 2021-03-22 2021-03-08  ...
                       2021-07-02 2021-06-18 2021-06-03 2021-05-20 2021-05-06 2021-04-19 2021-04-02 2021-03-19  ...
            datetime                                                                                            ...
            2018-01-02   0.079782   0.115975   0.070866   0.028849  -0.081170   0.140380   0.063864   0.110987  ...
            2018-01-03   0.123386   0.107789   0.071037   0.045278  -0.060782   0.167446   0.089779   0.124476  ...
            2018-01-04   0.140775   0.097206   0.063702   0.042415  -0.078164   0.173218   0.098914   0.114389  ...
            2018-01-05   0.030320  -0.037209  -0.044536  -0.047267  -0.081888   0.045648   0.059947   0.047652  ...
            2018-01-08   0.107201   0.009219  -0.015995  -0.036594  -0.086633   0.108965   0.122164   0.108508  ...
            ...               ...        ...        ...        ...        ...        ...        ...        ...  ...

        """

        # 1) prepare the prediction of proxy models
        perf_task_tpl = deepcopy(self.task_tpl)  # this task is supposed to contains no complicated objects

        trainer = auto_filter_kwargs(trainer)(experiment_name=self.exp_name, **trainer_kwargs)
        # NOTE:
        # The handler is initialized for only once.
        if not trainer.has_worker():
            self.dh = init_task_handler(perf_task_tpl)
            self.dh.config(dump_all=False)  # in some cases, the data handler are saved to disk with `dump_all=True`
        else:
            self.dh = init_instance_by_config(perf_task_tpl["dataset"]["kwargs"]["handler"])
        assert self.dh.dump_all is False  # otherwise, it will save all the detailed data

        seg = perf_task_tpl["dataset"]["kwargs"]["segments"]

        # We want to split the training time period into small segments.
        perf_task_tpl["dataset"]["kwargs"]["segments"] = {
            "train": (DatasetH.get_min_time(seg), DatasetH.get_max_time(seg)),
            "test": (None, None),
        }

        # NOTE:
        # we play a trick here
        # treat the training segments as test to create the rolling tasks
        rg = RollingGen(step=self.step, test_key="train", train_key=None, task_copy_func=deepcopy_basic_type)
        gen_task = task_generator(perf_task_tpl, [rg])

        recorders = R.list_recorders(experiment_name=self.exp_name)
        if len(gen_task) == len(recorders):
            get_module_logger("Internal Data").info("the data has been initialized")
        else:
            # train new models
            assert 0 == len(recorders), "An empty experiment is required for setup `InternalData`"
            trainer.train(gen_task)

        # 2) extract the similarity matrix
        label_df = self.dh.fetch(col_set="label")
        # for
        recorders = R.list_recorders(experiment_name=self.exp_name)

        key_l = []
        ic_l = []
        for _, rec in tqdm(recorders.items(), desc="calc"):
            pred = rec.load_object("pred.pkl")
            task = rec.load_object("task")
            data_key = task["dataset"]["kwargs"]["segments"]["train"]
            key_l.append(data_key)
            ic_l.append(delayed(self._calc_perf)(pred.iloc[:, 0], label_df.iloc[:, 0]))

        ic_l = Parallel(n_jobs=-1)(ic_l)
        self.data_ic_df = pd.DataFrame(dict(zip(key_l, ic_l)))
        self.data_ic_df = self.data_ic_df.sort_index().sort_index(axis=1)

        del self.dh  # handler is not useful now

    def _calc_perf(self, pred, label):
        df = pd.DataFrame({"pred": pred, "label": label})
        df = df.groupby("datetime").corr(method="spearman")
        corr = df.loc(axis=0)[:, "pred"]["label"].droplevel(axis=0, level=-1)
        return corr

    def update(self):
        """update the data for online trading"""
        # TODO:
        # when new data are totally(including label) available
        # - update the prediction
        # - update the data similarity map(if applied)


class MetaTaskDS(MetaTask):
    """Meta Task for Data Selection"""

    def __init__(self, task: dict, meta_info: pd.DataFrame, mode: str = MetaTask.PROC_MODE_FULL, fill_method="max"):
        """

        The description of the processed data

            time_perf: A array with shape  <hist_step_n * step, data pieces>  ->  data piece performance

            time_belong:  A array with shape <sample, data pieces>  -> belong or not (1. or 0.)
            array([[1., 0., 0., ..., 0., 0., 0.],
                   [1., 0., 0., ..., 0., 0., 0.],
                   [1., 0., 0., ..., 0., 0., 0.],
                   ...,
                   [0., 0., 0., ..., 0., 0., 1.],
                   [0., 0., 0., ..., 0., 0., 1.],
                   [0., 0., 0., ..., 0., 0., 1.]])

        Parameters
        ----------
        meta_info: pd.DataFrame
            please refer to the docs of _prepare_meta_ipt for detailed explanation.
        """
        super().__init__(task, meta_info)
        self.fill_method = fill_method

        time_perf = self._get_processed_meta_info()
        self.processed_meta_input = {"time_perf": time_perf}
        # FIXME: memory issue in this step
        if mode == MetaTask.PROC_MODE_FULL:
            # process metainfo_
            ds = self.get_dataset()

            # these three lines occupied 70% of the time of initializing MetaTaskDS
            d_train, d_test = ds.prepare(["train", "test"], col_set=["feature", "label"])
            prev_size = d_test.shape[0]
            d_train = d_train.dropna(axis=0)
            d_test = d_test.dropna(axis=0)
            if prev_size == 0 or d_test.shape[0] / prev_size <= 0.1:
                raise ValueError(f"Most of samples are dropped. Please check this task: {task}")

            assert (
                d_test.groupby("datetime").size().shape[0] >= 5
            ), "In this segment, this trading dates is less than 5, you'd better check the data."

            sample_time_belong = np.zeros((d_train.shape[0], time_perf.shape[1]))
            for i, col in enumerate(time_perf.columns):
                # these two lines of code occupied 20% of the time of initializing MetaTaskDS
                slc = slice(*d_train.index.slice_locs(start=col[0], end=col[1]))
                sample_time_belong[slc, i] = 1.0

            # If you want that last month also belongs to the last time_perf
            # Assumptions: the latest data has similar performance like the last month
            sample_time_belong[sample_time_belong.sum(axis=1) != 1, -1] = 1.0

            self.processed_meta_input.update(
                dict(
                    X=d_train["feature"],
                    y=d_train["label"].iloc[:, 0],
                    X_test=d_test["feature"],
                    y_test=d_test["label"].iloc[:, 0],
                    time_belong=sample_time_belong,
                    test_idx=d_test["label"].index,
                )
            )

        # TODO: set device: I think this is not necessary to converting data format.
        self.processed_meta_input = data_to_tensor(self.processed_meta_input)

    def _get_processed_meta_info(self):
        meta_info_norm = self.meta_info.sub(self.meta_info.mean(axis=1), axis=0)
        if self.fill_method.startswith("max"):
            suffix = self.fill_method.lstrip("max")
            if suffix == "seg":
                fill_value = {}
                for col in meta_info_norm.columns:
                    fill_value[col] = meta_info_norm.loc[meta_info_norm[col].isna(), :].dropna(axis=1).mean().max()
                fill_value = pd.Series(fill_value).sort_index()
                # The NaN Values are filled segment-wise. Below is an exampleof fill_value
                # 2009-01-05  2009-02-06    0.145809
                # 2009-02-09  2009-03-06    0.148005
                # 2009-03-09  2009-04-03    0.090385
                # 2009-04-07  2009-05-05    0.114318
                # 2009-05-06  2009-06-04    0.119328
                # ...
                meta_info_norm = meta_info_norm.fillna(fill_value)
            else:
                if len(suffix) > 0:
                    get_module_logger("MetaTaskDS").warning(
                        f"fill_method={self.fill_method}; the info after can't be correctly parsed. Please check your parameters."
                    )
                fill_value = meta_info_norm.max(axis=1)
                # fill it with row max to align with previous implementation
                # This will magnify the data similarity when data is in daily freq

                # the fill value corresponds to data like this
                # It get a performance value for each day.
                # The performance value are get from other models on this day
                # 2009-01-16    0.276320
                # 2009-01-19    0.280603
                #                 ...
                # 2011-06-27    0.203773
                meta_info_norm = meta_info_norm.T.fillna(fill_value).T
        elif self.fill_method == "zero":
            # It will fillna(0.0) at the end.
            pass
        else:
            raise NotImplementedError(f"This type of input is not supported")
        meta_info_norm = meta_info_norm.fillna(0.0)  # always fill zero in case of NaN
        return meta_info_norm

    def get_meta_input(self):
        return self.processed_meta_input


class MetaDatasetDS(MetaTaskDataset):
    def __init__(
        self,
        *,
        task_tpl: Union[dict, list],
        step: int,
        trunc_days: int = None,
        rolling_ext_days: int = 0,
        exp_name: Union[str, InternalData],
        segments: Union[Dict[Text, Tuple], float],
        hist_step_n: int = 10,
        task_mode: str = MetaTask.PROC_MODE_FULL,
        fill_method: str = "max",
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

        exp_name : Union[str, InternalData]
            Decide what meta_info are used for prediction.
            - str: the name of the experiment to store the performance of data
            - InternalData: a prepared internal data
        segments: Union[Dict[Text, Tuple], float]
            the segments to divide data
            both left and right
            if segments is a float:
                the float represents the percentage of data for training
        hist_step_n: int
            length of historical steps for the meta infomation
        task_mode : str
            Please refer to the docs of MetaTask
        """
        super().__init__(segments=segments)
        if isinstance(exp_name, InternalData):
            self.internal_data = exp_name
        else:
            self.internal_data = InternalData(task_tpl, step=step, exp_name=exp_name)
            self.internal_data.setup()
        self.task_tpl = deepcopy(task_tpl)  # FIXME: if the handler is shared, how to avoid the explosion of the memroy.
        self.trunc_days = trunc_days
        self.hist_step_n = hist_step_n
        self.step = step

        if isinstance(task_tpl, dict):
            rg = RollingGen(
                step=step, trunc_days=trunc_days, task_copy_func=deepcopy_basic_type
            )  # NOTE: trunc_days is very important !!!!
            task_iter = rg(task_tpl)
            if rolling_ext_days > 0:
                self.ta = TimeAdjuster(future=True)
                for t in task_iter:
                    t["dataset"]["kwargs"]["segments"]["test"] = self.ta.shift(
                        t["dataset"]["kwargs"]["segments"]["test"], step=rolling_ext_days, rtype=RollingGen.ROLL_EX
                    )
            if task_mode == MetaTask.PROC_MODE_FULL:
                # Only pre initializing the task when full task is req
                # initializing handler and share it.
                init_task_handler(task_tpl)
        else:
            assert isinstance(task_tpl, list)
            task_iter = task_tpl

        self.task_list = []
        self.meta_task_l = []
        logger = get_module_logger("MetaDatasetDS")
        logger.info(f"Example task for training meta model: {task_iter[0]}")
        for t in tqdm(task_iter, desc="creating meta tasks"):
            try:
                self.meta_task_l.append(
                    MetaTaskDS(t, meta_info=self._prepare_meta_ipt(t), mode=task_mode, fill_method=fill_method)
                )
                self.task_list.append(t)
            except ValueError as e:
                logger.warning(f"ValueError: {e}")
        assert len(self.meta_task_l) > 0, "No meta tasks found. Please check the data and setting"

    def _prepare_meta_ipt(self, task) -> pd.DataFrame:
        """
        Please refer to `self.internal_data.setup` for detailed information about `self.internal_data.data_ic_df`

        Indices with format below can be successfully sliced by  `ic_df.loc[:end, pd.IndexSlice[:, :end]]`

               2021-06-21 2021-06-04 .. 2021-03-22 2021-03-08
               2021-07-02 2021-06-18 .. 2021-04-02 None

        Returns
        -------
            a pd.DataFrame with similar content below.
            - each column corresponds to a trained model named by the training data range
            - each row corresponds to a day of data tested by the models of the columns
            - The rows cells that overlaps with the data used by columns are masked


                       2009-01-05 2009-02-09 ... 2011-04-27 2011-05-26
                       2009-02-06 2009-03-06 ... 2011-05-25 2011-06-23
            datetime                         ...
            2009-01-13        NaN   0.310639 ...  -0.169057   0.137792
            2009-01-14        NaN   0.261086 ...  -0.143567   0.082581
            ...               ...        ... ...        ...        ...
            2011-06-30  -0.054907  -0.020219 ...  -0.023226        NaN
            2011-07-01  -0.075762  -0.026626 ...  -0.003167        NaN

        """
        ic_df = self.internal_data.data_ic_df

        segs = task["dataset"]["kwargs"]["segments"]
        end = max(segs[k][1] for k in ("train", "valid") if k in segs)
        ic_df_avail = ic_df.loc[:end, pd.IndexSlice[:, :end]]

        # meta data set focus on the **information** instead of preprocess
        # 1) filter the overlap info
        def mask_overlap(s):
            """
            mask overlap information
            data after self.name[end] with self.trunc_days that contains future info are also considered as overlap info

            Approximately the diagnal + horizon length of data are masked.
            """
            start, end = s.name
            end = get_date_by_shift(trading_date=end, shift=self.trunc_days - 1, future=True)
            return s.mask((s.index >= start) & (s.index <= end))

        ic_df_avail = ic_df_avail.apply(mask_overlap)  # apply to each col

        # 2) filter the info with too long periods
        total_len = self.step * self.hist_step_n
        if ic_df_avail.shape[0] >= total_len:
            return ic_df_avail.iloc[-total_len:]
        else:
            raise ValueError("the history of distribution data is not long enough.")

    def _prepare_seg(self, segment: Text) -> List[MetaTask]:
        if isinstance(self.segments, float):
            train_task_n = int(len(self.meta_task_l) * self.segments)
            if segment == "train":
                return self.meta_task_l[:train_task_n]
            elif segment == "test":
                return self.meta_task_l[train_task_n:]
            else:
                raise NotImplementedError(f"This type of input is not supported")
        else:
            raise NotImplementedError(f"This type of input is not supported")
