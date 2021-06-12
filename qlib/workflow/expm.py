# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.entities import ViewType
import os, logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Text

from .exp import MLflowExperiment, Experiment
from ..config import C
from .recorder import Recorder
from ..log import get_module_logger

logger = get_module_logger("workflow", logging.INFO)


class ExpManager:
    """
    This is the `ExpManager` class for managing experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """

    def __init__(self, uri: Text, default_exp_name: Optional[Text]):
        self._current_uri = uri
        self._default_exp_name = default_exp_name
        self.active_experiment = None  # only one experiment can active each time

    def __repr__(self):
        return "{name}(current_uri={curi})".format(name=self.__class__.__name__, curi=self._current_uri)

    def start_exp(
        self,
        *,
        experiment_id: Optional[Text] = None,
        experiment_name: Optional[Text] = None,
        recorder_id: Optional[Text] = None,
        recorder_name: Optional[Text] = None,
        uri: Optional[Text] = None,
        resume: bool = False,
        **kwargs,
    ):
        """
        Start an experiment. This method includes first get_or_create an experiment, and then
        set it to be active.

        Parameters
        ----------
        experiment_id : str
            id of the active experiment.
        experiment_name : str
            name of the active experiment.
        recorder_id : str
            id of the recorder to be started.
        recorder_name : str
            name of the recorder to be started.
        uri : str
            the current tracking URI.
        resume : boolean
            whether to resume the experiment and recorder.

        Returns
        -------
        An active experiment.
        """
        raise NotImplementedError(f"Please implement the `start_exp` method.")

    def end_exp(self, recorder_status: Text = Recorder.STATUS_S, **kwargs):
        """
        End an active experiment.

        Parameters
        ----------
        experiment_name : str
            name of the active experiment.
        recorder_status : str
            the status of the active recorder of the experiment.
        """
        raise NotImplementedError(f"Please implement the `end_exp` method.")

    def create_exp(self, experiment_name: Optional[Text] = None):
        """
        Create an experiment.

        Parameters
        ----------
        experiment_name : str
            the experiment name, which must be unique.

        Returns
        -------
        An experiment object.
        """
        raise NotImplementedError(f"Please implement the `create_exp` method.")

    def search_records(self, experiment_ids=None, **kwargs):
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

    def get_exp(self, *, experiment_id=None, experiment_name=None, create: bool = True, start: bool = False):
        """
        Retrieve an experiment. This method includes getting an active experiment, and get_or_create a specific experiment.

        When user specify experiment id and name, the method will try to return the specific experiment.
        When user does not provide recorder id or name, the method will try to return the current active experiment.
        The `create` argument determines whether the method will automatically create a new experiment according
        to user's specification if the experiment hasn't been created before.

        * If `create` is True:

            * If `active experiment` exists:

                * no id or name specified, return the active experiment.
                * if id or name is specified, return the specified experiment. If no such exp found, create a new experiment with given id or name. If `start` is set to be True, the experiment is set to be active.

            * If `active experiment` not exists:

                * no id or name specified, create a default experiment.
                * if id or name is specified, return the specified experiment. If no such exp found, create a new experiment with given id or name. If `start` is set to be True, the experiment is set to be active.

        * Else If `create` is False:

            * If `active experiment` exists:

                * no id or name specified, return the active experiment.
                * if id or name is specified, return the specified experiment. If no such exp found, raise Error.

            * If `active experiment` not exists:

                *  no id or name specified. If the default experiment exists, return it, otherwise, raise Error.
                * if id or name is specified, return the specified experiment. If no such exp found, raise Error.

        Parameters
        ----------
        experiment_id : str
            id of the experiment to return.
        experiment_name : str
            name of the experiment to return.
        create : boolean
            create the experiment it if hasn't been created before.
        start : boolean
            start the new experiment if one is created.

        Returns
        -------
        An experiment object.
        """
        # special case of getting experiment
        if experiment_id is None and experiment_name is None:
            if self.active_experiment is not None:
                return self.active_experiment
            # User don't want get active code now.
            experiment_name = self._default_exp_name

        if create:
            exp, is_new = self._get_or_create_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        else:
            exp, is_new = (
                self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name),
                False,
            )
        if is_new and start:
            self.active_experiment = exp
            # start the recorder
            self.active_experiment.start()
        return exp

    def _get_or_create_exp(self, experiment_id=None, experiment_name=None) -> (object, bool):
        """
        Method for getting or creating an experiment. It will try to first get a valid experiment, if exception occurs, it will
        automatically create a new experiment based on the given id and name.
        """
        try:
            return (
                self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name),
                False,
            )
        except ValueError:
            if experiment_name is None:
                experiment_name = self._default_exp_name
            logger.warning(f"No valid experiment found. Create a new experiment with name {experiment_name}.")
            return self.create_exp(experiment_name), True

    def _get_exp(self, experiment_id=None, experiment_name=None) -> Experiment:
        """
        Get specific experiment by name or id. If it does not exist, raise ValueError.

        Parameters
        ----------
        experiment_id :
            The id of experiment
        experiment_name :
            The name of experiment

        Returns
        -------
        Experiment:
            The searched experiment

        Raises
        ------
        ValueError
        """
        raise NotImplementedError(f"Please implement the `_get_exp` method")

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

    @property
    def default_uri(self):
        """
        Get the default tracking URI from qlib.config.C
        """
        if "kwargs" not in C.exp_manager or "uri" not in C.exp_manager["kwargs"]:
            raise ValueError("The default URI is not set in qlib.config.C")
        return C.exp_manager["kwargs"]["uri"]

    @property
    def uri(self):
        """
        Get the default tracking URI or current URI.

        Returns
        -------
        The tracking URI string.
        """
        return self._current_uri or self.default_uri

    def set_uri(self, uri: Optional[Text] = None):
        """
        Set the current tracking URI and the corresponding variables.

        Parameters
        ----------
        uri  : str

        """
        if uri is None:
            logger.info("No tracking URI is provided. Use the default tracking URI.")
            self._current_uri = self.default_uri
        else:
            # Temporarily re-set the current uri as the uri argument.
            self._current_uri = uri
        # Customized features for subclasses.
        self._set_uri()

    def _set_uri(self):
        """
        Customized features for subclasses' set_uri function.
        """
        raise NotImplementedError(f"Please implement the `_set_uri` method.")

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

    def __init__(self, uri: Text, default_exp_name: Optional[Text]):
        super(MLflowExpManager, self).__init__(uri, default_exp_name)
        self._client = None

    def _set_uri(self):
        self._client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        logger.info("{:}".format(self._client))

    @property
    def client(self):
        # Delay the creation of mlflow client in case of creating `mlruns` folder when importing qlib
        if self._client is None:
            self._client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        return self._client

    def start_exp(
        self,
        *,
        experiment_id: Optional[Text] = None,
        experiment_name: Optional[Text] = None,
        recorder_id: Optional[Text] = None,
        recorder_name: Optional[Text] = None,
        uri: Optional[Text] = None,
        resume: bool = False,
    ):
        # Set the tracking uri
        self.set_uri(uri)
        # Create experiment
        if experiment_name is None:
            experiment_name = self._default_exp_name
        experiment, _ = self._get_or_create_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        # Set up active experiment
        self.active_experiment = experiment
        # Start the experiment
        self.active_experiment.start(recorder_id=recorder_id, recorder_name=recorder_name, resume=resume)

        return self.active_experiment

    def end_exp(self, recorder_status: Text = Recorder.STATUS_S):
        if self.active_experiment is not None:
            self.active_experiment.end(recorder_status)
            self.active_experiment = None
        # When an experiment end, we will release the current uri.
        self._current_uri = None

    def create_exp(self, experiment_name: Optional[Text] = None):
        assert experiment_name is not None
        # init experiment
        experiment_id = self.client.create_experiment(experiment_name)
        experiment = MLflowExperiment(experiment_id, experiment_name, self.uri)
        experiment._default_name = self._default_exp_name

        return experiment

    def _get_exp(self, experiment_id=None, experiment_name=None):
        """
        Method for getting or creating an experiment. It will try to first get a valid experiment, if exception occurs, it will
        raise errors.
        """
        assert (
            experiment_id is not None or experiment_name is not None
        ), "Please input at least one of experiment/recorder id or name before retrieving experiment/recorder."
        if experiment_id is not None:
            try:
                # NOTE: the mlflow's experiment_id must be str type...
                # https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.get_experiment
                exp = self.client.get_experiment(experiment_id)
                if exp.lifecycle_stage.upper() == "DELETED":
                    raise MlflowException("No valid experiment has been found.")
                experiment = MLflowExperiment(exp.experiment_id, exp.name, self.uri)
                return experiment
            except MlflowException:
                raise ValueError(
                    "No valid experiment has been found, please make sure the input experiment id is correct."
                )
        elif experiment_name is not None:
            try:
                exp = self.client.get_experiment_by_name(experiment_name)
                if exp is None or exp.lifecycle_stage.upper() == "DELETED":
                    raise MlflowException("No valid experiment has been found.")
                experiment = MLflowExperiment(exp.experiment_id, experiment_name, self.uri)
                return experiment
            except MlflowException as e:
                raise ValueError(
                    "No valid experiment has been found, please make sure the input experiment name is correct."
                )

    def search_records(self, experiment_ids, **kwargs):
        filter_string = "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        run_view_type = 1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        max_results = 100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        order_by = kwargs.get("order_by")
        return self.client.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by)

    def delete_exp(self, experiment_id=None, experiment_name=None):
        assert (
            experiment_id is not None or experiment_name is not None
        ), "Please input a valid experiment id or name before deleting."
        try:
            if experiment_id is not None:
                self.client.delete_experiment(experiment_id)
            else:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    raise MlflowException("No valid experiment has been found.")
                self.client.delete_experiment(experiment.experiment_id)
        except MlflowException as e:
            raise Exception(
                f"Error: {e}. Something went wrong when deleting experiment. Please check if the name/id of the experiment is correct."
            )

    def list_experiments(self):
        # retrieve all the existing experiments
        exps = self.client.list_experiments(view_type=ViewType.ACTIVE_ONLY)
        experiments = dict()
        for exp in exps:
            experiment = MLflowExperiment(exp.experiment_id, exp.name, self.uri)
            experiments[exp.name] = experiment
        return experiments
