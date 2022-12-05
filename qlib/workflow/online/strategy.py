# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineStrategy module is an element of online serving.
"""

from typing import List, Union
from qlib.log import get_module_logger
from qlib.model.ens.group import RollingGroup
from qlib.utils import transform_end_date
from qlib.workflow.online.utils import OnlineTool, OnlineToolR
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.collect import Collector, RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.utils import TimeAdjuster


class OnlineStrategy:
    """
    OnlineStrategy is working with `Online Manager <#Online Manager>`_, responding to how the tasks are generated, the models are updated and signals are prepared.
    """

    def __init__(self, name_id: str):
        """
        Init OnlineStrategy.
        This module **MUST** use `Trainer <../reference/api.html#qlib.model.trainer.Trainer>`_ to finishing model training.

        Args:
            name_id (str): a unique name or id.
            trainer (qlib.model.trainer.Trainer, optional): a instance of Trainer. Defaults to None.
        """
        self.name_id = name_id
        self.logger = get_module_logger(self.__class__.__name__)
        self.tool = OnlineTool()

    def prepare_tasks(self, cur_time, **kwargs) -> List[dict]:
        """
        After the end of a routine, check whether we need to prepare and train some new tasks based on cur_time (None for latest)..
        Return the new tasks waiting for training.

        You can find the last online models by OnlineTool.online_models.
        """
        raise NotImplementedError(f"Please implement the `prepare_tasks` method.")

    def prepare_online_models(self, trained_models, cur_time=None) -> List[object]:
        """
        Select some models from trained models and set them to online models.
        This is a typical implementation to online all trained models, you can override it to implement the complex method.
        You can find the last online models by OnlineTool.online_models if you still need them.

        NOTE: Reset all online models to trained models. If there are no trained models, then do nothing.

        **NOTE**:
            Current implementation is very naive. Here is a more complex situation which is more closer to the
            practical scenarios.
            1. Train new models at the day before `test_start` (at time stamp `T`)
            2. Switch models at the `test_start` (at time timestamp `T + 1` typically)

        Args:
            models (list): a list of models.
            cur_time (pd.Dataframe): current time from OnlineManger. None for the latest.

        Returns:
            List[object]: a list of online models.
        """
        if not trained_models:
            return self.tool.online_models()
        self.tool.reset_online_tag(trained_models)
        return trained_models

    def first_tasks(self) -> List[dict]:
        """
        Generate a series of tasks firstly and return them.
        """
        raise NotImplementedError(f"Please implement the `first_tasks` method.")

    def get_collector(self) -> Collector:
        """
        Get the instance of `Collector <../advanced/task_management.html#Task Collecting>`_ to collect different results of this strategy.

        For example:
            1) collect predictions in Recorder
            2) collect signals in a txt file

        Returns:
            Collector
        """
        raise NotImplementedError(f"Please implement the `get_collector` method.")


class RollingStrategy(OnlineStrategy):

    """
    This example strategy always uses the latest rolling model sas online models.
    """

    def __init__(
        self,
        name_id: str,
        task_template: Union[dict, List[dict]],
        rolling_gen: RollingGen,
    ):
        """
        Init RollingStrategy.

        Assumption: the str of name_id, the experiment name, and the trainer's experiment name are the same.

        Args:
            name_id (str): a unique name or id. Will be also the name of the Experiment.
            task_template (Union[dict, List[dict]]): a list of task_template or a single template, which will be used to generate many tasks using rolling_gen.
            rolling_gen (RollingGen): an instance of RollingGen
        """
        super().__init__(name_id=name_id)
        self.exp_name = self.name_id
        if not isinstance(task_template, list):
            task_template = [task_template]
        self.task_template = task_template
        self.rg = rolling_gen
        assert issubclass(self.rg.__class__, RollingGen), "The rolling strategy relies on the feature if RollingGen"
        self.tool = OnlineToolR(self.exp_name)
        self.ta = TimeAdjuster()

    def get_collector(self, process_list=[RollingGroup()], rec_key_func=None, rec_filter_func=None, artifacts_key=None):
        """
        Get the instance of `Collector <../advanced/task_management.html#Task Collecting>`_ to collect results. The returned collector must distinguish results in different models.

        Assumption: the models can be distinguished based on the model name and rolling test segments.
        If you do not want this assumption, please implement your method or use another rec_key_func.

        Args:
            rec_key_func (Callable): a function to get the key of a recorder. If None, use recorder id.
            rec_filter_func (Callable, optional): filter the recorder by return True or False. Defaults to None.
            artifacts_key (List[str], optional): the artifacts key you want to get. If None, get all artifacts.
        """

        def rec_key(recorder):
            task_config = recorder.load_object("task")
            model_key = task_config["model"]["class"]
            rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            return model_key, rolling_key

        if rec_key_func is None:
            rec_key_func = rec_key

        artifacts_collector = RecorderCollector(
            experiment=self.exp_name,
            process_list=process_list,
            rec_key_func=rec_key_func,
            rec_filter_func=rec_filter_func,
            artifacts_key=artifacts_key,
        )

        return artifacts_collector

    def first_tasks(self) -> List[dict]:
        """
        Use rolling_gen to generate different tasks based on task_template.

        Returns:
            List[dict]: a list of tasks
        """
        return task_generator(
            tasks=self.task_template,
            generators=self.rg,  # generate different date segment
        )

    def prepare_tasks(self, cur_time) -> List[dict]:
        """
        Prepare new tasks based on cur_time (None for the latest).

        You can find the last online models by OnlineToolR.online_models.

        Returns:
            List[dict]: a list of new tasks.
        """
        # TODO: filter recorders by latest test segments is not a necessary
        latest_records, max_test = self._list_latest(self.tool.online_models())
        if max_test is None:
            self.logger.warn(f"No latest online recorders, no new tasks.")
            return []
        calendar_latest = transform_end_date(cur_time)
        self.logger.info(
            f"The interval between current time {calendar_latest} and last rolling test begin time {max_test[0]} is {self.ta.cal_interval(calendar_latest, max_test[0])}, the rolling step is {self.rg.step}"
        )
        res = []
        for rec in latest_records:
            task = rec.load_object("task")
            res.extend(self.rg.gen_following_tasks(task, calendar_latest))
        return res

    def _list_latest(self, rec_list: List[Recorder]):
        """
        List latest recorder form rec_list

        Args:
            rec_list (List[Recorder]): a list of Recorder

        Returns:
            List[Recorder], pd.Timestamp: the latest recorders and their test end time
        """
        if len(rec_list) == 0:
            return rec_list, None
        max_test = max(rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"] for rec in rec_list)
        latest_rec = []
        for rec in rec_list:
            if rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"] == max_test:
                latest_rec.append(rec)
        return latest_rec, max_test
