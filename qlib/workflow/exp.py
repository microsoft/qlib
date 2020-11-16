# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
from datetime import datetime
from pathlib import Path
from .recorder import Recorder, MLflowRecorder
from ..log import get_module_logger

logger = get_module_logger("workflow", "INFO")


class Experiment:
    """
    Thie is the `Experiment` class for each experiment being run. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.active_recorder = None  # only one recorder can running each time

    def __repr__(self):
        return str(self.info)

    def __str__(self):
        return str(self.info)

    @property
    def info(self):
        recorders = self.list_recorders()
        output = dict()
        output["class"] = "Experiment"
        output["id"] = self.id
        output["name"] = self.name
        output["active_recorder"] = self.active_recorder.id if self.active_recorder is not None else None
        output["recorders"] = list(recorders.keys())
        return output

    def start(self):
        """
        Start the experiment and set it to be active. This method will also start a new recorder.

        Returns
        -------
        An active recorder.
        """
        raise NotImplementedError(f"Please implement the `start` method.")

    def end(self, recorder_status=Recorder.STATUS_S):
        """
        End the experiment.

        Parameters
        ----------
        recorder_status : str
            the status the recorder to be set with when ending (SCHEDULED, RUNNING, FINISHED, FAILED).
        """
        raise NotImplementedError(f"Please implement the `end` method.")

    def create_recorder(self):
        """
        Create a recorder for each experiment.

        Returns
        -------
        A recorder object.
        """
        raise NotImplementedError(f"Please implement the `create_recorder` method.")

    def search_records(self, **kwargs):
        """
        Get a pandas DataFrame of records that fit the search criteria of the experiment.
        Inputs are the search critera user want to apply.

        Returns
        -------
        A pandas.DataFrame of records, where each metric, parameter, and tag
        are expanded into their own columns named metrics.*, params.*, and tags.*
        respectively. For records that don't have a particular metric, parameter, or tag, their
        value will be (NumPy) Nan, None, or None respectively.
        """
        raise NotImplementedError(f"Please implement the `search_records` method.")

    def delete_recorder(self, recorder_id):
        """
        Create a recorder for each experiment.

        Parameters
        ----------
        recorder_id : str
            the id of the recorder to be deleted.
        """
        raise NotImplementedError(f"Please implement the `delete_recorder` method.")

    def get_recorder(self, recorder_id=None, recorder_name=None, create: bool = True):
        """
        Retrieve a Recorder for user. When user specify recorder id and name, the method will try to return the
        specific recorder. When user does not provide recorder id or name, the method will try to return the current
        active recorder. The `create` argument determines whether the method will automatically create a new recorder
        according to user's specification if the recorder hasn't been created before

        If `create` is True:
            If R's running:
                1) no id or name specified, return the active recorder.
                2) if id or name is specified, return the specified recorder. If no such exp found,
                create a new recorder with given id or name, and the recorder shoud be running.
            If R's not running:
                1) no id or name specified, create a new recorder.
                2) if id or name is specified, return the specified experiment. If no such exp found,
                create a new recorder with given id or name, and the recorder shoud be running.
        Else If `create` is False:
            If R's running:
                1) no id or name specified, return the active recorder.
                2) if id or name is specified, return the specified recorder. If no such exp found,
                raise Error.
            If R's not running:
                1) no id or name specified, raise Error.
                2) if id or name is specified, return the specified recorder. If no such exp found,
                raise Error.

        Parameters
        ----------
        recorder_id : str
            the id of the recorder to be deleted.
        recorder_name : str
            the name of the recorder to be deleted.

        Returns
        -------
        A recorder object.
        """
        raise NotImplementedError(f"Please implement the `get_recorder` method.")

    def list_recorders(self):
        """
        List all the existing recorders of this experiment.

        Returns
        -------
        A dictionary (id -> recorder) of recorder information that being stored.
        """
        raise NotImplementedError(f"Please implement the `list_recorders` method.")


class MLflowExperiment(Experiment):
    """
    Use mlflow to implement Experiment.
    """

    def __init__(self, id, name, uri):
        super(MLflowExperiment, self).__init__(id, name)
        self._uri = uri
        self._default_name = None
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)

    def start(self):
        # set the active experiment
        mlflow.set_experiment(self.name)
        logger.info(f"Experiment {self.id} starts running ...")
        # set up recorder
        recorder = self.create_recorder()
        self.active_recorder = recorder
        # start the recorder
        run = self.active_recorder.start_run()

        return self.active_recorder

    def end(self, recorder_status):
        if self.active_recorder is not None:
            self.active_recorder.end_run(recorder_status)
            self.active_recorder = None

    def create_recorder(self):
        recorders = self.list_recorders()
        num = len(recorders)
        name = "Recorder_{}".format(num + 1)
        recorder = MLflowRecorder(name, self.id, self._uri)

        return recorder

    def search_records(self, **kwargs):
        filter_string = "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        run_view_type = 1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        max_results = 100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        order_by = kwargs.get("order_by")

        return self.client.search_runs([self.id], filter_string, run_view_type, max_results, order_by)

    def delete_recorder(self, recorder_id=None, recorder_name=None):
        assert (
            recorder_id is not None or recorder_name is not None
        ), "Please input a valid recorder id or name before deleting."
        try:
            if recorder_id is not None:
                self.client.delete_run(recorder_id)
            else:
                recorders = self.list_recorders()
                for r in recorders:
                    if recorders[r].name == recorder_name:
                        recorder_id = r
                        break
                self.client.delete_run(recorder_id)
        except:
            raise Exception(
                "Something went wrong when deleting recorder. Please check if the name/id of the recorder is correct."
            )

    def get_recorder(self, recorder_id=None, recorder_name=None, create=True):
        """
        MLflow doesn't support create recorder with a specific id. Thus, when user only provides recorder id and `create`
        is set to True, this method will not automatically create an active recorder.
        """
        # retrive all the recorders under this experiment
        recorders = self.list_recorders()
        if recorder_id is None and recorder_name is None:
            if self.active_recorder:
                return self.active_recorder
            else:
                if create:
                    self.start()
                    logger.warning(
                        f"Recorder {self.active_recorder.id} is running under the experiment with name {self.name}..."
                    )
                    return self.active_recorder
                else:
                    raise Exception(
                        "Something went wrong when retrieving recorders. Please check if QlibRecorder is running or the name/id of the recorder is correct."
                    )
        else:
            if recorder_id is not None:
                if recorder_id in recorders:
                    return recorders[recorder_id]
                else:
                    # mlflow does not support create a run with given id
                    raise Exception(
                        "Something went wrong when retrieving recorders. Please check if QlibRecorder is running or the name/id of the recorder is correct."
                    )
            else:
                for rid in recorders:
                    if recorders[rid].name == recorder_name:
                        return recorders[rid]
                if create:
                    recorders = self.list_recorders()
                    logger.warning(f"No valid recorder found. Create a new recorder with name {recorder_name}.")
                    recorder = self.create_recorder()
                    recorder.name = recorder_name
                    recorder.start_run()
                    return recorder
                else:
                    raise Exception(
                        "Something went wrong when retrieving experiments. Please check if QlibRecorder is running or the name/id of the experiment is correct."
                    )

    def list_recorders(self):
        runs = self.client.list_run_infos(self.id, run_view_type=1)[::-1]
        recorders = dict()
        for i in range(len(runs)):
            rid = runs[i].run_id
            status = runs[i].status
            start_time = runs[i].start_time
            end_time = runs[i].end_time
            recorder = MLflowRecorder(f"Recorder_{i+1}", self.id, self._uri)
            recorder.id = rid
            recorder.status = status
            recorder.start_time = (
                datetime.fromtimestamp(float(start_time) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                if start_time is not None
                else None
            )
            recorder.end_time = (
                datetime.fromtimestamp(float(end_time) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                if end_time is not None
                else None
            )
            recorder._uri = self._uri
            recorders[rid] = recorder

        return recorders
