# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OnlineTool is a module to set and unset a series of `online` models.
The `online` models are some decisive models in some time points, which can be changed with the change of time.
This allows us to use efficient submodels as the market-style changing.
"""

from typing import List, Union
from qlib.data.dataset import TSDatasetH

from qlib.log import get_module_logger
from qlib.utils import get_cls_kwargs
from qlib.utils.exceptions import LoadObjectError
from qlib.workflow.online.update import PredUpdater
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.utils import list_recorders


class OnlineTool:
    """
    OnlineTool will manage `online` models in an experiment that includes the model recorders.
    """

    ONLINE_KEY = "online_status"  # the online status key in recorder
    ONLINE_TAG = "online"  # the 'online' model
    OFFLINE_TAG = "offline"  # the 'offline' model, not for online serving

    def __init__(self):
        """
        Init OnlineTool.
        """
        self.logger = get_module_logger(self.__class__.__name__)

    def set_online_tag(self, tag, recorder: Union[list, object]):
        """
        Set `tag` to the model to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `OFFLINE_TAG`
            recorder (Union[list,object]): the model's recorder
        """
        raise NotImplementedError(f"Please implement the `set_online_tag` method.")

    def get_online_tag(self, recorder: object) -> str:
        """
        Given a model recorder and return its online tag.

        Args:
            recorder (Object): the model's recorder

        Returns:
            str: the online tag
        """
        raise NotImplementedError(f"Please implement the `get_online_tag` method.")

    def reset_online_tag(self, recorder: Union[list, object]):
        """
        Offline all models and set the recorders to 'online'.

        Args:
            recorder (Union[list,object]):
                the recorder you want to reset to 'online'.

        """
        raise NotImplementedError(f"Please implement the `reset_online_tag` method.")

    def online_models(self) -> list:
        """
        Get current `online` models

        Returns:
            list: a list of `online` models.
        """
        raise NotImplementedError(f"Please implement the `online_models` method.")

    def update_online_pred(self, to_date=None):
        """
        Update the predictions of `online` models to to_date.

        Args:
            to_date (pd.Timestamp): the pred before this date will be updated. None for updating to the latest.

        """
        raise NotImplementedError(f"Please implement the `update_online_pred` method.")


class OnlineToolR(OnlineTool):
    """
    The implementation of OnlineTool based on (R)ecorder.
    """

    def __init__(self, default_exp_name: str = None):
        """
        Init OnlineToolR.

        Args:
            default_exp_name (str): the default experiment name.
        """
        super().__init__()
        self.default_exp_name = default_exp_name

    def set_online_tag(self, tag, recorder: Union[Recorder, List]):
        """
        Set `tag` to the model's recorder to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `NEXT_ONLINE_TAG`, `OFFLINE_TAG`
            recorder (Union[Recorder, List]): a list of Recorder or an instance of Recorder
        """
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        for rec in recorder:
            rec.set_tags(**{self.ONLINE_KEY: tag})
        self.logger.info(f"Set {len(recorder)} models to '{tag}'.")

    def get_online_tag(self, recorder: Recorder) -> str:
        """
        Given a model recorder and return its online tag.

        Args:
            recorder (Recorder): an instance of recorder

        Returns:
            str: the online tag
        """
        tags = recorder.list_tags()
        return tags.get(self.ONLINE_KEY, self.OFFLINE_TAG)

    def reset_online_tag(self, recorder: Union[Recorder, List], exp_name: str = None):
        """
        Offline all models and set the recorders to 'online'.

        Args:
            recorder (Union[Recorder, List]):
                the recorder you want to reset to 'online'.
            exp_name (str): the experiment name. If None, then use default_exp_name.

        """
        exp_name = self._get_exp_name(exp_name)
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        recs = list_recorders(exp_name)
        self.set_online_tag(self.OFFLINE_TAG, list(recs.values()))
        self.set_online_tag(self.ONLINE_TAG, recorder)

    def online_models(self, exp_name: str = None) -> list:
        """
        Get current `online` models

        Args:
            exp_name (str): the experiment name. If None, then use default_exp_name.

        Returns:
            list: a list of `online` models.
        """
        exp_name = self._get_exp_name(exp_name)
        return list(list_recorders(exp_name, lambda rec: self.get_online_tag(rec) == self.ONLINE_TAG).values())

    def update_online_pred(self, to_date=None, exp_name: str = None):
        """
        Update the predictions of online models to to_date.

        Args:
            to_date (pd.Timestamp): the pred before this date will be updated. None for updating to latest time in Calendar.
            exp_name (str): the experiment name. If None, then use default_exp_name.
        """
        exp_name = self._get_exp_name(exp_name)
        online_models = self.online_models(exp_name=exp_name)
        for rec in online_models:
            hist_ref = 0
            task = rec.load_object("task")
            # Special treatment of historical dependencies
            cls, kwargs = get_cls_kwargs(task["dataset"], default_module="qlib.data.dataset")
            if issubclass(cls, TSDatasetH):
                hist_ref = kwargs.get("step_len", TSDatasetH.DEFAULT_STEP_LEN)
            try:
                updater = PredUpdater(rec, to_date=to_date, hist_ref=hist_ref)
            except LoadObjectError as e:
                # skip the recorder without pred
                self.logger.warn(f"An exception `{str(e)}` happened when load `pred.pkl`, skip it.")
                continue
            updater.update()

        self.logger.info(f"Finished updating {len(online_models)} online model predictions of {exp_name}.")

    def _get_exp_name(self, exp_name):
        if exp_name is None:
            if self.default_exp_name is None:
                raise ValueError(
                    "Both default_exp_name and exp_name are None. OnlineToolR needs a specific experiment."
                )
            exp_name = self.default_exp_name
        return exp_name
