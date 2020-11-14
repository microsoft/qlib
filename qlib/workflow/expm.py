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
    This is the `ExpManager` class for managing experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """

    def __init__(self, uri, default_exp_name):
        self.uri = uri
        self.default_exp_name = default_exp_name
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

        Returns
        -------
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

    def get_exp(self, experiment_id=None, experiment_name=None, create: bool = True):
        """
        Retrieve an experiment by experiment_id from the backend store.

        Parameters
        ----------
        experiment_id : str
            the experiment id to return.
        create : boolean
            create the experiment if it does not exists

        Returns
        -------
        An experiment object.
        """
        raise NotImplementedError(f"Please implement the `get_exp` method.")

    def delete_exp(self, experiment_id=None, experiment_name=None):
        """
        Delete an experiment.

        Parameters
        ----------
        experiment_id  : str
            the experiment id.
        experiment_name  : str
            the experiment name.
        """
        raise NotImplementedError(f"Please implement the `delete_exp` method.")

    def get_uri(self):
        """
        Get the default tracking URI or current URI.

        Returns
        -------
        The tracking URI string.
        """
        return self.uri

    def list_experiments(self):
        """
        List all the existing experiments.

        Returns
        -------
        A dictionary (name -> experiment) of experiments information that being stored.
        """
        raise NotImplementedError(f"Please implement the `list_experiments` method.")


class MLflowExpManager(ExpManager):
    """
    Use mlflow to implement ExpManager.
    """

    def __init__(self, uri, default_exp_name):
        super(MLflowExpManager, self).__init__(uri, default_exp_name)
        self._total_exps = 0
        # get all the exps
        self.experiments = self.list_experiments()

    def start_exp(self, experiment_name=None, uri=None):
        # create experiment
        experiment = self.create_exp(experiment_name, uri)
        # set up active experiment
        self.active_experiment = experiment
        # start the experiment
        self.active_experiment.start()
        self._total_exps += 1  # update exp num

        return self.active_experiment

    def end_exp(self, status):
        if self.active_experiment is not None:
            self.active_experiment.end(status)
            self.active_experiment = None
            self._total_exps -= 1

    def create_exp(self, experiment_name=None, uri=None):
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
            logger.info(
                f"No experiment name provided. The default experiment name is set as `{self.default_exp_name}`."
            )
            experiment_id = mlflow.create_experiment(self.default_exp_name)
            # set the active experiment
            mlflow.set_experiment(self.default_exp_name)
            experiment_name = self.default_exp_name
        else:
            if experiment_name not in self.experiments:
                if mlflow.get_experiment_by_name(experiment_name) is not None:
                    logger.info(
                        "The experiment has already been created before. Try to resume the experiment with a new recorder..."
                    )
                    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
                else:
                    experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = self.experiments[experiment_name].id
                experiment = self.experiments[experiment_name]
        # set the active experiment
        mlflow.set_experiment(experiment_name)
        # init experiment
        experiment = MLflowExperiment(experiment_id, experiment_name, self.uri)
        experiment._default_name = self.default_exp_name
        # store the experiment
        self.experiments[experiment_name] = experiment

        return experiment

    def search_records(self, experiment_ids, **kwargs):
        filter_string = "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        run_view_type = 1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        max_results = 100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        order_by = kwargs.get("order_by")
        return mlflow.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by)

    def get_exp(self, experiment_id=None, experiment_name=None, create=True):
        if experiment_id is None and experiment_name is None:
            if self.active_experiment:
                return self.active_experiment
            else:
                if create:
                    logger.warning("QlibRecorder is not running. Use the Default experiment for further process.")
                    return self.start_exp()
                else:
                    raise Exception(
                        "Something went wrong when retrieving experiments. Please check if QlibRecorder is running or the name/id of the experiment is correct."
                    )
        else:
            if experiment_name is not None:
                if experiment_name in self.experiments:
                    return self.experiments[experiment_name]
                else:
                    if create:
                        logger.warning(
                            f"No valid experiment found. Create experiment with name {experiment_name} for further process."
                        )
                        return self.start_exp(experiment_name)
                    else:
                        raise Exception(
                            "Something went wrong when retrieving experiments. Please check if QlibRecorder is running or the name/id of the experiment is correct."
                        )
            else:
                for name in self.experiments:
                    if self.experiments[name].id == experiment_id:
                        return self.experiments[name]
                if create:
                    logger.warning(f"No valid experiment found. Use the Default experiment for further process.")
                    return self.start_exp()
                else:
                    raise Exception(
                        "Something went wrong when retrieving experiments. Please check if QlibRecorder is running or the name/id of the experiment is correct."
                    )

    def delete_exp(self, experiment_id=None, experiment_name=None):
        assert (
            experiment_id is not None or experiment_name is not None
        ), "Please input a valid experiment id or name before deleting."
        try:
            if experiment_id is not None:
                mlflow.delete_experiment(experiment_id)
                self.experiments = {key: val for key, val in self.experiments.items() if val.id != experiment_id}
            else:
                experiment_id = self.experiments[experiment_name].id
                mlflow.delete_experiment(experiment_id)
        except:
            raise Exception(
                "Something went wrong when deleting experiment. Please check if the name/id of the experiment is correct."
            )

    def list_experiments(self):
        # retrieve all the existing experiments
        client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        exps = client.list_experiments()
        experiments = dict()
        self._total_exps = len(exps)
        for i in range(len(exps)):
            eid = exps[i].experiment_id
            ename = exps[i].name
            experiment = MLflowExperiment(eid, ename, self.uri)
            experiment.id = eid
            experiment.name = ename
            experiment._uri = self.uri
            experiments[ename] = experiment

        return experiments
