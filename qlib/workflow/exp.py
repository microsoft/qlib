# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union
import mlflow, logging
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from pathlib import Path
from .recorder import Recorder, MLflowRecorder
from ..log import get_module_logger

logger = get_module_logger("workflow", logging.INFO)


class Experiment:
    """
    This is the `Experiment` class for each experiment being run. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.active_recorder = None  # only one recorder can running each time

    def __repr__(self):
        return "{name}(id={id}, info={info})".format(name=self.__class__.__name__, id=self.id, info=self.info)

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

    def start(self, *, recorder_id=None, recorder_name=None, resume=False):
        """
        Start the experiment and set it to be active. This method will also start a new recorder.

        Parameters
        ----------
        recorder_id : str
            the id of the recorder to be created.
        recorder_name : str
            the name of the recorder to be created.
        resume : bool
            whether to resume the first recorder

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

    def create_recorder(self, recorder_name=None):
        """
        Create a recorder for each experiment.

        Parameters
        ----------
        recorder_name : str
            the name of the recorder to be created.

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

    def get_recorder(self, recorder_id=None, recorder_name=None, create: bool = True, start: bool = False):
        """
        Retrieve a Recorder for user. When user specify recorder id and name, the method will try to return the
        specific recorder. When user does not provide recorder id or name, the method will try to return the current
        active recorder. The `create` argument determines whether the method will automatically create a new recorder
        according to user's specification if the recorder hasn't been created before.

        * If `create` is True:

            * If `active recorder` exists:

                * no id or name specified, return the active recorder.
                * if id or name is specified, return the specified recorder. If no such exp found, create a new recorder with given id or name. If `start` is set to be True, the recorder is set to be active.

            * If `active recorder` not exists:

                * no id or name specified, create a new recorder.
                * if id or name is specified, return the specified experiment. If no such exp found, create a new recorder with given id or name. If `start` is set to be True, the recorder is set to be active.

        * Else If `create` is False:

            * If `active recorder` exists:

                * no id or name specified, return the active recorder.
                * if id or name is specified, return the specified recorder. If no such exp found, raise Error.

            * If `active recorder` not exists:

                * no id or name specified, raise Error.
                * if id or name is specified, return the specified recorder. If no such exp found, raise Error.

        Parameters
        ----------
        recorder_id : str
            the id of the recorder to be deleted.
        recorder_name : str
            the name of the recorder to be deleted.
        create : boolean
            create the recorder if it hasn't been created before.
        start : boolean
            start the new recorder if one is created.

        Returns
        -------
        A recorder object.
        """
        # special case of getting the recorder
        if recorder_id is None and recorder_name is None:
            if self.active_recorder is not None:
                return self.active_recorder
            recorder_name = self._default_rec_name
        if create:
            recorder, is_new = self._get_or_create_rec(recorder_id=recorder_id, recorder_name=recorder_name)
        else:
            recorder, is_new = (
                self._get_recorder(recorder_id=recorder_id, recorder_name=recorder_name),
                False,
            )
        if is_new and start:
            self.active_recorder = recorder
            # start the recorder
            self.active_recorder.start_run()
        return recorder

    def _get_or_create_rec(self, recorder_id=None, recorder_name=None) -> (object, bool):
        """
        Method for getting or creating a recorder. It will try to first get a valid recorder, if exception occurs, it will
        automatically create a new recorder based on the given id and name.
        """
        try:
            if recorder_id is None and recorder_name is None:
                recorder_name = self._default_rec_name
            return (
                self._get_recorder(recorder_id=recorder_id, recorder_name=recorder_name),
                False,
            )
        except ValueError:
            if recorder_name is None:
                recorder_name = self._default_rec_name
            logger.info(f"No valid recorder found. Create a new recorder with name {recorder_name}.")
            return self.create_recorder(recorder_name), True

    def _get_recorder(self, recorder_id=None, recorder_name=None):
        """
        Get specific recorder by name or id. If it does not exist, raise ValueError

        Parameters
        ----------
        recorder_id :
            The id of recorder
        recorder_name :
            The name of recorder

        Returns
        -------
        Recorder:
            The searched recorder

        Raises
        ------
        ValueError
        """
        raise NotImplementedError(f"Please implement the `_get_recorder` method")

    def list_recorders(self, **flt_kwargs):
        """
        List all the existing recorders of this experiment. Please first get the experiment instance before calling this method.
        If user want to use the method `R.list_recorders()`, please refer to the related API document in `QlibRecorder`.

        flt_kwargs : dict
            filter recorders by conditions
            e.g.  list_recorders(status=Recorder.STATUS_FI)

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
        self._default_rec_name = "mlflow_recorder"
        self._client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)

    def __repr__(self):
        return "{name}(id={id}, info={info})".format(name=self.__class__.__name__, id=self.id, info=self.info)

    def start(self, *, recorder_id=None, recorder_name=None, resume=False):
        logger.info(f"Experiment {self.id} starts running ...")
        # Get or create recorder
        if recorder_name is None:
            recorder_name = self._default_rec_name
        # resume the recorder
        if resume:
            recorder, _ = self._get_or_create_rec(recorder_id=recorder_id, recorder_name=recorder_name)
        # create a new recorder
        else:
            recorder = self.create_recorder(recorder_name)
        # Set up active recorder
        self.active_recorder = recorder
        # Start the recorder
        self.active_recorder.start_run()

        return self.active_recorder

    def end(self, recorder_status):
        if self.active_recorder is not None:
            self.active_recorder.end_run(recorder_status)
            self.active_recorder = None

    def create_recorder(self, recorder_name=None):
        if recorder_name is None:
            recorder_name = self._default_rec_name
        recorder = MLflowRecorder(self.id, self._uri, recorder_name)

        return recorder

    def _get_recorder(self, recorder_id=None, recorder_name=None):
        """
        Method for getting or creating a recorder. It will try to first get a valid recorder, if exception occurs, it will
        raise errors.
        """
        assert (
            recorder_id is not None or recorder_name is not None
        ), "Please input at least one of recorder id or name before retrieving recorder."
        if recorder_id is not None:
            try:
                run = self._client.get_run(recorder_id)
                recorder = MLflowRecorder(self.id, self._uri, mlflow_run=run)
                return recorder
            except MlflowException:
                raise ValueError("No valid recorder has been found, please make sure the input recorder id is correct.")
        elif recorder_name is not None:
            logger.warning(
                f"Please make sure the recorder name {recorder_name} is unique, we will only return the latest recorder if there exist several matched the given name."
            )
            recorders = self.list_recorders()
            for rid in recorders:
                if recorders[rid].name == recorder_name:
                    return recorders[rid]
            raise ValueError("No valid recorder has been found, please make sure the input recorder name is correct.")

    def search_records(self, **kwargs):
        filter_string = "" if kwargs.get("filter_string") is None else kwargs.get("filter_string")
        run_view_type = 1 if kwargs.get("run_view_type") is None else kwargs.get("run_view_type")
        max_results = 100000 if kwargs.get("max_results") is None else kwargs.get("max_results")
        order_by = kwargs.get("order_by")

        return self._client.search_runs([self.id], filter_string, run_view_type, max_results, order_by)

    def delete_recorder(self, recorder_id=None, recorder_name=None):
        assert (
            recorder_id is not None or recorder_name is not None
        ), "Please input a valid recorder id or name before deleting."
        try:
            if recorder_id is not None:
                self._client.delete_run(recorder_id)
            else:
                recorder = self._get_recorder(recorder_name=recorder_name)
                self._client.delete_run(recorder.id)
        except MlflowException as e:
            raise Exception(
                f"Error: {e}. Something went wrong when deleting recorder. Please check if the name/id of the recorder is correct."
            )

    UNLIMITED = 50000  # FIXME: Mlflow can only list 50000 records at most!!!!!!!

    def list_recorders(self, max_results: int = UNLIMITED, status: Union[str, None] = None):
        """
        Parameters
        ----------
        max_results : int
            the number limitation of the results
        status : str
            the criteria based on status to filter results.
            `None` indicates no filtering.
        """
        runs = self._client.search_runs(self.id, run_view_type=ViewType.ACTIVE_ONLY, max_results=max_results)
        recorders = dict()
        for i in range(len(runs)):
            recorder = MLflowRecorder(self.id, self._uri, mlflow_run=runs[i])
            if status is None or recorder.status == status:
                recorders[runs[i].info.run_id] = recorder

        return recorders
