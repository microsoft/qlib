# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
from pathlib import Path
from .recorder import MLflowRecorder
from ..log import get_module_logger

logger = get_module_logger("workflow", "INFO")


class Experiment:
    """
    Thie is the `Experiment` class for each experiment being run. The API is designed
    """

    def __init__(self):
        self.name = None
        self.id = None
        self.active_recorder = None  # only one recorder can running each time
        self.recorders = dict()  # recorder id -> object

    def __repr__(self):
        return str(self.info)

    def __str__(self):
        return str(self.info)

    @property
    def info(self):
        output = dict()
        output["class"] = "Experiment"
        output["id"] = self.id
        output["name"] = self.name
        output["active_recorder"] = self.active_recorder.id
        output["recorders"] = list(self.recorders.keys())

    def start(self):
        """
        Start the experiment.

        Parameters
        ----------

        Returns
        -------
        A running recorder instance.
        """
        raise NotImplementedError(f"Please implement the `start` method.")

    def end(self, status):
        """
        End the experiment.

        Parameters
        ----------
        status : str
            the status the recorder to be set with when ending (SCHEDULED, RUNNING, FINISHED, FAILED).
        """
        raise NotImplementedError(f"Please implement the `end` method.")

    def create_recorder(self):
        """
        Create a recorder for each experiment.

        Parameters
        ----------

        Returns
        -------
        A recorder object.
        """
        raise NotImplementedError(f"Please implement the `create_recorder` method.")

    def search_records(self, **kwargs):
        """
        Get a pandas DataFrame of records that fit the search criteria of the experiment.

        Parameters
        ----------
        filter_string : str
            filter query string, defaults to searching all runs.
        run_view_type : int
            one of enum values ACTIVE_ONLY, DELETED_ONLY, or ALL (e.g. in mlflow.entities.ViewType).
        max_results  : int
            the maximum number of runs to put in the dataframe.
        order_by : list
            list of columns to order by (e.g., “metrics.rmse”).

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

    def get_recorder(self, recorder_id=None, recorder_name=None):
        """
        Get the current active Recorder.

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


class MLflowExperiment(Experiment):
    """
    Use mlflow to implement Experiment.
    """

    def start(self):
        # set up recorder
        recorder = self.create_recorder()
        self.active_recorder = recorder
        # start the recorder
        run = self.active_recorder.start_run()
        # store the recorder
        self.recorders[self.active_recorder.id] = recorder
        return self.active_recorder

    def end(self, status):
        if self.active_recorder is not None:
            self.active_recorder.end_run(status)
            self.active_recorder = None

    def create_recorder(self):
        num = len(self.recorders)
        name = "Recorder_{}".format(num + 1)
        recorder = MLflowRecorder(name, self.id)
        return recorder

    def search_records(self, **kwargs):
        filter_string = "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        run_view_type = 1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        max_results = 100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        order_by = kwargs.get("order_by")
        return mlflow.search_runs([self.id], filter_string, run_view_type, max_results, order_by)

    def delete_recorder(self, recorder_id):
        mlflow.delete_run(recorder_id)
        self.recorders = [r for r in self.recorders if r.id == recorder_id]

    def get_recorder(self, recorder_id=None, recorder_name=None):
        if recorder_id is not None:
            return self.recorders[recorder_id]
        elif recorder_name is not None:
            for rid in self.recorders:
                if self.recorders[rid].name == recorder_name:
                    return self.recorders[rid]
        elif self.active_recorder is None:
            raise Exception("No valid active recorder exists. Please make sure the experiment is running.")
        else:
            logger.info("No experiment id or name is given. Return the current active experiment.")
            return self.active_recorder
