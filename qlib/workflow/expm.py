# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from urllib.parse import urlparse
import mlflow
from filelock import FileLock
from mlflow.exceptions import MlflowException, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.entities import ViewType
import os
from typing import Optional, Text

from .exp import MLflowExperiment, Experiment
from ..config import C
from .recorder import Recorder
from ..log import get_module_logger
from ..utils.exceptions import ExpAlreadyExistError


logger = get_module_logger("workflow")


class ExpManager:
    """
    This is the `ExpManager` class for managing experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)

    The `ExpManager` is expected to be a singleton (btw, we can have multiple `Experiment`s with different uri. user can get different experiments from different uri, and then compare records of them). Global Config (i.e. `C`)  is also a singleton.

    So we try to align them together.  They share the same variable, which is called **default uri**. Please refer to `ExpManager.default_uri` for details of variable sharing.

    When the user starts an experiment, the user may want to set the uri to a specific uri (it will override **default uri** during this period), and then unset the **specific uri** and fallback to the **default uri**.    `ExpManager._active_exp_uri` is that **specific uri**.
    """

    active_experiment: Optional[Experiment]

    def __init__(self, uri: Text, default_exp_name: Optional[Text]):
        self.default_uri = uri
        self._active_exp_uri = None  # No active experiments. So it is set to None
        self._default_exp_name = default_exp_name
        self.active_experiment = None  # only one experiment can be active each time
        logger.debug(f"experiment manager uri is at {self.uri}")

    def __repr__(self):
        return "{name}(uri={uri})".format(name=self.__class__.__name__, uri=self.uri)

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
    ) -> Experiment:
        """
        Start an experiment. This method includes first get_or_create an experiment, and then
        set it to be active.

        Maintaining `_active_exp_uri` is included in start_exp, remaining implementation should be included in _end_exp in subclass

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
        self._active_exp_uri = uri
        # The subclass may set the underlying uri back.
        # So setting `_active_exp_uri` come before `_start_exp`
        return self._start_exp(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            recorder_id=recorder_id,
            recorder_name=recorder_name,
            resume=resume,
            **kwargs,
        )

    def _start_exp(self, *args, **kwargs) -> Experiment:
        """Please refer to the doc of `start_exp`"""
        raise NotImplementedError(f"Please implement the `start_exp` method.")

    def end_exp(self, recorder_status: Text = Recorder.STATUS_S, **kwargs):
        """
        End an active experiment.

        Maintaining `_active_exp_uri` is included in end_exp, remaining implementation should be included in _end_exp in subclass

        Parameters
        ----------
        experiment_name : str
            name of the active experiment.
        recorder_status : str
            the status of the active recorder of the experiment.
        """
        self._active_exp_uri = None
        # The subclass may set the underlying uri back.
        # So setting `_active_exp_uri` come before `_end_exp`
        self._end_exp(recorder_status=recorder_status, **kwargs)

    def _end_exp(self, recorder_status: Text = Recorder.STATUS_S, **kwargs):
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

        Raise
        -----
        ExpAlreadyExistError
        """
        raise NotImplementedError(f"Please implement the `create_exp` method.")

    def search_records(self, experiment_ids=None, **kwargs):
        """
        Get a pandas DataFrame of records that fit the search criteria of the experiment.
        Inputs are the search criteria user want to apply.

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
            exp, _ = self._get_or_create_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        else:
            exp = self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        if self.active_experiment is None and start:
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

            # NOTE: mlflow doesn't consider the lock for recording multiple runs
            # So we supported it in the interface wrapper
            pr = urlparse(self.uri)
            if pr.scheme == "file":
                with FileLock(os.path.join(pr.netloc, pr.path, "filelock")):  # pylint: disable=E0110
                    return self.create_exp(experiment_name), True
            # NOTE: for other schemes like http, we double check to avoid create exp conflicts
            try:
                return self.create_exp(experiment_name), True
            except ExpAlreadyExistError:
                return (
                    self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name),
                    False,
                )

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

    @default_uri.setter
    def default_uri(self, value):
        C.exp_manager.setdefault("kwargs", {})["uri"] = value

    @property
    def uri(self):
        """
        Get the default tracking URI or current URI.

        Returns
        -------
        The tracking URI string.
        """
        return self._active_exp_uri or self.default_uri

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

    @property
    def client(self):
        # Please refer to `tests/dependency_tests/test_mlflow.py::MLflowTest::test_creating_client`
        # The test ensure the speed of create a new client
        return mlflow.tracking.MlflowClient(tracking_uri=self.uri)

    def _start_exp(
        self,
        *,
        experiment_id: Optional[Text] = None,
        experiment_name: Optional[Text] = None,
        recorder_id: Optional[Text] = None,
        recorder_name: Optional[Text] = None,
        resume: bool = False,
    ):
        # Create experiment
        if experiment_name is None:
            experiment_name = self._default_exp_name
        experiment, _ = self._get_or_create_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        # Set up active experiment
        self.active_experiment = experiment
        # Start the experiment
        self.active_experiment.start(recorder_id=recorder_id, recorder_name=recorder_name, resume=resume)

        return self.active_experiment

    def _end_exp(self, recorder_status: Text = Recorder.STATUS_S):
        if self.active_experiment is not None:
            self.active_experiment.end(recorder_status)
            self.active_experiment = None

    def create_exp(self, experiment_name: Optional[Text] = None):
        assert experiment_name is not None
        # init experiment
        try:
            experiment_id = self.client.create_experiment(experiment_name)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                raise ExpAlreadyExistError() from e
            raise e

        return MLflowExperiment(experiment_id, experiment_name, self.uri)

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
            except MlflowException as e:
                raise ValueError(
                    "No valid experiment has been found, please make sure the input experiment id is correct."
                ) from e
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
                ) from e

    def search_records(self, experiment_ids=None, **kwargs):
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
            raise ValueError(
                f"Error: {e}. Something went wrong when deleting experiment. Please check if the name/id of the experiment is correct."
            ) from e

    def list_experiments(self):
        # retrieve all the existing experiments
        exps = self.client.list_experiments(view_type=ViewType.ACTIVE_ONLY)
        experiments = dict()
        for exp in exps:
            experiment = MLflowExperiment(exp.experiment_id, exp.name, self.uri)
            experiments[exp.name] = experiment
        return experiments
