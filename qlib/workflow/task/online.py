from typing import Union, List
from qlib import get_module_logger
from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.collect import TaskCollector
from qlib.workflow.task.update import ModelUpdater


class OnlineManagement:
    def __init__(self, experiment_name):
        pass

    def update_online_pred(self, recorder: Union[str, Recorder]):
        """update the predictions of online models

        Parameters
        ----------
        recorder : Union[str, Recorder]
            the id or the instance of Recorder

        """
        raise NotImplementedError(f"Please implement the `update_pred` method.")

    def prepare_new_models(self, tasks: List[dict]):
        """prepare(train) new models

        Parameters
        ----------
        tasks : List[dict]
            a list of tasks

        """
        raise NotImplementedError(f"Please implement the `prepare_new_models` method.")

    def reset_online_model(self, recorders: List[Union[str, Recorder]]):
        """reset online model

        Parameters
        ----------
        recorders : List[Union[str, Recorder]]
            a list of the recorder id or the instance

        """
        raise NotImplementedError(f"Please implement the `reset_online_model` method.")


class RollingOnlineManager(OnlineManagement):

    ONLINE_TAG = "online_model"
    ONLINE_TAG_TRUE = "True"
    ONLINE_TAG_FALSE = "False"

    def __init__(self, experiment_name: str) -> None:
        """ModelUpdater needs experiment name to find the records

        Parameters
        ----------
        experiment_name : str
            experiment name string
        """
        super(RollingOnlineManager, self).__init__(experiment_name)
        self.logger = get_module_logger("RollingOnlineManager")
        self.exp_name = experiment_name
        self.tc = TaskCollector(experiment_name)

    def set_online_model(self, recorder: Union[str, Recorder]):
        """online model will be identified at the tags of the record

        Parameters
        ----------
        recorder: Union[str,Recorder]
            the id of a Recorder or the Recorder instance
        """
        if isinstance(recorder, str):
            recorder = self.tc.get_recorder_by_id(recorder_id=recorder)
        recorder.set_tags(**{self.ONLINE_TAG: self.ONLINE_TAG_TRUE})

    def cancel_online_model(self, recorder: Union[str, Recorder]):
        if isinstance(recorder, str):
            recorder = self.tc.get_recorder_by_id(recorder_id=recorder)
        recorder.set_tags(**{self.ONLINE_TAG: self.ONLINE_TAG_FALSE})

    def cancel_all_online_model(self):
        recs = self.tc.list_recorders()
        for rid, rec in recs.items():
            self.cancel_online_model(rec)

    def reset_online_model(self, recorders: Union[str, List[Union[str, Recorder]]]):
        """cancel all online model and reset the given model to online model

        Parameters
        ----------
        recorders: List[Union[str,Recorder]]
            the list of the id of a Recorder or the Recorder instance
        """
        self.cancel_all_online_model()
        if isinstance(recorders, str):
            recorders = [recorders]
        for rec_or_rid in recorders:
            self.set_online_model(rec_or_rid)

    def online_filter(self, recorder):
        tags = recorder.list_tags()
        if tags.get(self.ONLINE_TAG, self.ONLINE_TAG_FALSE) == self.ONLINE_TAG_TRUE:
            return True
        return False

    def list_online_model(self):
        """list the record of online model

        Returns
        -------
        dict
            {rid : recorder of the online model}
        """

        return self.tc.list_recorders(rec_filter_func=self.online_filter)

    def update_online_pred(self):
        """update all online model predictions to the latest day in Calendar."""
        mu = ModelUpdater(self.exp_name)
        cnt = mu.update_all_pred(self.online_filter)
        self.logger.info(f"Finish updating {cnt} online model predictions of {self.exp_name}.")
