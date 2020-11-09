# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
import os
from pathlib import Path
from contextlib import contextmanager
from .exp import MLflowExperiment
from .recorder import MLflowRecorder
from ..log import get_module_logger

logger = get_module_logger("workflow", "INFO")


class ExpManager:
    """
    This is the `ExpManager` class for managing the experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """

    def __init__(self):
        self.uri = None
        self.active_experiment = None  # only one experiment can running each time
        self.experiments = dict()  # store the experiment name --> Experiment object

    def start_exp(self, experiment_name=None, uri=None, **kwargs):
        """
        Start running an experiment.

        Parameters
        ----------
        experiment_name : str
            name of the active experiment.
        uri : str
            the current tracking URI.
        artifact_location : str
            the location to store all the artifacts.
        nested : boolean
            controls whether run is nested in parent run.

        Returns
        An active recorder.
        """
        raise NotImplementedError(f"Please implement the `start_exp` method.")

    def end_exp(self, **kwargs):
        """
        End an running experiment.

        Parameters
        ----------
        experiment_name : str
            name of the active experiment.
        """
        raise NotImplementedError(f"Please implement the `end_exp` method.")

    def search_records(self, experiment_ids=None, **kwargs):
        """
        Get a pandas DataFrame of records that fit the search criteria.

        Parameters
        ----------
        experiment_ids : list
            list of experiment IDs.
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

    def create_exp(self, experiment_name, artifact_location=None):
        """
        Create an experiment.

        Parameters
        ----------
        experiment_name : str
            the experiment name, which must be unique.
        artifact_location : str
            the location to store run artifacts.

        Returns
        -------
        An experiment object.
        """
        raise NotImplementedError(f"Please implement the `create_exp` method.")

    def get_exp(self, experiment_id=None, experiment_name=None):
        """
        Retrieve an experiment by experiment_id from the backend store.

        Parameters
        ----------
        experiment_id : str
            the experiment id to return.

        Returns
        -------
        An experiment object.
        """
        raise NotImplementedError(f"Please implement the `get_exp` method.")

    def delete_exp(self, experiment_id):
        """
        Delete an experiment.

        Parameters
        ----------
        experiment_id  : str
            the experiment id.
        """
        raise NotImplementedError(f"Please implement the `create_exp` method.")

    def get_uri(self):
        """
        Get the default tracking URI or current URI.

        Parameters
        ----------

        Returns
        -------
        The tracking URI string.
        """
        return self.uri


class MLflowExpManager(ExpManager):
    """
    Use mlflow to implement ExpManager.
    """

    def __init__(self):
        super(MLflowExpManager, self).__init__()
        self.uri = None

    def start_exp(self, experiment_name=None, uri=None):
        # create experiment
        experiment = self.create_exp(experiment_name, uri)
        # set up active experiment
        self.active_experiment = experiment
        # store the experiment
        self.experiments[experiment_name] = experiment
        # start the experiment
        self.active_experiment.start()

        return self.active_experiment

    def end_exp(self, status):
        if self.active_experiment is not None:
            self.active_experiment.end(status)
            self.active_experiment = None

    def create_exp(self, experiment_name=None, uri=None):
        # init experiment
        experiment = MLflowExperiment()
        # set the tracking uri
        if uri is None:
            logger.info(
                "No tracking URI is provided. The default tracking URI is set as `mlruns` under the working directory."
            )
        else:
            self.uri = uri
        mlflow.set_tracking_uri(self.uri)
        # start the experiment
        if experiment_name is None:
            logger.info("No experiment name provided. The default experiment name is set as `experiment`.")
            experiment_id = mlflow.create_experiment("experiment")
            # set the active experiment
            mlflow.set_experiment("experiment")
            experiment_name = "experiment"
        else:
            if experiment_name not in self.experiments:
                if mlflow.get_experiment_by_name(experiment_name) is not None:
                    logger.info(
                        "The experiment has already been created before. Try to resume the experiment..."
                    )
                    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
                else:
                    experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = self.experiments[experiment_name].id
                experiment = self.experiments[experiment_name]
        # set the active experiment
        mlflow.set_experiment(experiment_name)
        # set up experiment
        experiment.id = experiment_id
        experiment.name = experiment_name

        return experiment

    def search_records(self, experiment_ids, **kwargs):
        filter_string = "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        run_view_type = 1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        max_results = 100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        order_by = kwargs.get("order_by")
        return mlflow.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by)

    def get_exp(self, experiment_id=None, experiment_name=None):
        if experiment_name is not None:
            return self.experiments[experiment_name]
        elif experiment_id is not None:
            for name in self.experiments:
                if self.experiments[name].id == experiment_id:
                    return self.experiments[name]
        elif self.active_experiment is None:
            raise Exception('No valid active experiment exists. Please make sure experiment manager is running.')
        else:
            logger.info(
                "No experiment id or name is given. Return the current active experiment."
            )
            return self.active_experiment

    def delete_exp(self, experiment_id):
        mlflow.delete_experiment(experiment_id)
        self.experiments = {key: val for key, val in self.experiments.items() if val.id != experiment_id}
