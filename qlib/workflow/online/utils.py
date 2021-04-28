"""
This module is like a online backend, deciding which models are `online` models and how can change them
"""
from typing import List, Union
from qlib.log import get_module_logger
from qlib.workflow.online.update import PredUpdater
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.utils import list_recorders


class OnlineTool:

    ONLINE_KEY = "online_status"  # the online status key in recorder
    ONLINE_TAG = "online"  # the 'online' model
    # NOTE: The meaning of this tag is that we can not assume the training models can be trained before we need its predition. Whenever finished training, it can be guaranteed that there are some online models.
    NEXT_ONLINE_TAG = "next_online"  # the 'next online' model, which can be 'online' model when call reset_online_model
    OFFLINE_TAG = "offline"  # the 'offline' model, not for online serving

    def __init__(self, need_log=True):
        """
        init OnlineTool.

        Args:
            need_log (bool, optional): print log or not. Defaults to True.
        """
        self.logger = get_module_logger(self.__class__.__name__)
        self.need_log = need_log
        self.cur_time = None

    def set_online_tag(self, tag, recorder):
        """
        Set `tag` to the model to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `NEXT_ONLINE_TAG`, `OFFLINE_TAG`
        """
        raise NotImplementedError(f"Please implement the `set_online_tag` method.")

    def get_online_tag(self):
        """
        Given a model and return its online tag.
        """
        raise NotImplementedError(f"Please implement the `get_online_tag` method.")

    def reset_online_tag(self, recorders=None):
        """offline all models and set the recorders to 'online'. If no parameter and no 'next online' model, then do nothing.

        Args:
            recorders (List, optional):
                the recorders you want to reset to 'online'. If don't give, set 'next online' model to 'online' model. If there isn't any 'next online' model, then maintain existing 'online' model.

        Returns:
            list: new online recorder. [] if there is no update.
        """
        raise NotImplementedError(f"Please implement the `reset_online_tag` method.")

    def online_models(self):
        """
        Return `online` models.
        """
        raise NotImplementedError(f"Please implement the `online_models` method.")

    def update_online_pred(self, to_date=None):
        """
        Update the predictions of online models to a date.

        Args:
            to_date (pd.Timestamp): the pred before this date will be updated. None for latest.

        """
        raise NotImplementedError(f"Please implement the `update_online_pred` method.")


class OnlineToolR(OnlineTool):
    """
    The implementation of OnlineTool based on (R)ecorder.

    """

    def __init__(self, experiment_name: str, need_log=True):
        """
        init OnlineToolR.

        Args:
            experiment_name (str): the experiment name.
            need_log (bool, optional): print log or not. Defaults to True.
        """
        super().__init__(need_log=need_log)
        self.exp_name = experiment_name

    def set_online_tag(self, tag, recorder: Union[Recorder, List]):
        """
        Set `tag` to the model to sign whether online.

        Args:
            tag (str): the tags in `ONLINE_TAG`, `NEXT_ONLINE_TAG`, `OFFLINE_TAG`
            recorder (Union[Recorder, List])
        """
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        for rec in recorder:
            rec.set_tags(**{self.ONLINE_KEY: tag})
        if self.need_log:
            self.logger.info(f"Set {len(recorder)} models to '{tag}'.")

    def get_online_tag(self, recorder: Recorder):
        """
        Given a model and return its online tag.

        Args:
            recorder (Recorder): a instance of recorder

        Returns:
            str: the tag
        """
        tags = recorder.list_tags()
        return tags.get(self.ONLINE_KEY, self.OFFLINE_TAG)

    def reset_online_tag(self, recorder: Union[Recorder, List] = None):
        """offline all models and set the recorders to 'online'. If no parameter and no 'next online' model, then do nothing.

        Args:
            recorders (Union[Recorder, List], optional):
                the recorders you want to reset to 'online'. If don't give, set 'next online' model to 'online' model. If there isn't any 'next online' model, then maintain existing 'online' model.

        Returns:
            list: new online recorder. [] if there is no update.
        """
        if recorder is None:
            recorder = list(
                list_recorders(self.exp_name, lambda rec: self.get_online_tag(rec) == self.NEXT_ONLINE_TAG).values()
            )
        if isinstance(recorder, Recorder):
            recorder = [recorder]
        if len(recorder) == 0:
            if self.need_log:
                self.logger.info("No 'next online' model, just use current 'online' models.")
            return []
        recs = list_recorders(self.exp_name)
        self.set_online_tag(self.OFFLINE_TAG, list(recs.values()))
        self.set_online_tag(self.ONLINE_TAG, recorder)
        return recorder

    def online_models(self):
        """
        Return online models.

        Returns:
            list: the list of online models
        """
        return list(list_recorders(self.exp_name, lambda rec: self.get_online_tag(rec) == self.ONLINE_TAG).values())

    def update_online_pred(self, to_date=None):
        """
        Update the predictions of online models to a date.

        Args:
            to_date (pd.Timestamp): the pred before this date will be updated. None for latest in Calendar.
        """
        online_models = self.online_models()
        for rec in online_models:
            PredUpdater(rec, to_date=to_date, need_log=self.need_log).update()

        if self.need_log:
            self.logger.info(f"Finished updating {len(online_models)} online model predictions of {self.exp_name}.")
