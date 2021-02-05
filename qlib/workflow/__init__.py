# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from .expm import MLflowExpManager
from .exp import Experiment
from .recorder import Recorder
from ..utils import Wrapper


class QlibRecorder:
    """
    A global system that helps to manage the experiments.
    """

    def __init__(self, exp_manager):
        self.exp_manager = exp_manager

    @contextmanager
    def start(self, experiment_name=None, recorder_name=None):
        """
        Method to start an experiment. This method can only be called within a Python's `with` statement. Here is the example code:

        .. code-block:: Python

            with R.start('test', 'recorder_1'):
                model.fit(dataset)
                R.log...
                ... # further operations

        Parameters
        ----------
        experiment_name : str
            name of the experiment one wants to start.
        recorder_name : str
            name of the recorder under the experiment one wants to start.
        """
        run = self.start_exp(experiment_name, recorder_name)
        try:
            yield run
        except Exception as e:
            self.end_exp(Recorder.STATUS_FA)  # end the experiment if something went wrong
            raise e
        self.end_exp(Recorder.STATUS_FI)

    def start_exp(self, experiment_name=None, recorder_name=None, uri=None):
        """
        Lower level method for starting an experiment. When use this method, one should end the experiment manually
        and the status of the recorder may not be handled properly. Here is the example code:

        .. code-block:: Python

            R.start_exp(experiment_name='test', recorder_name='recorder_1')
            ... # further operations
            R.end_exp('FINISHED') or R.end_exp(Recorder.STATUS_S)


        Parameters
        ----------
        experiment_name : str
            the name of the experiment to be started
        recorder_name : str
            name of the recorder under the experiment one wants to start.
        uri : str
            the tracking uri of the experiment, where all the artifacts/metrics etc. will be stored.
            The default uri are set in the qlib.config.

        Returns
        -------
        An experiment instance being started.
        """
        return self.exp_manager.start_exp(experiment_name, recorder_name, uri)

    def end_exp(self, recorder_status=Recorder.STATUS_FI):
        """
        Method for ending an experiment manually. It will end the current active experiment, as well as its
        active recorder with the specified `status` type. Here is the example code of the method:

        .. code-block:: Python

            R.start_exp(experiment_name='test')
            ... # further operations
            R.end_exp('FINISHED') or R.end_exp(Recorder.STATUS_S)

        Parameters
        ----------
        status : str
            The status of a recorder, which can be SCHEDULED, RUNNING, FINISHED, FAILED.
        """
        self.exp_manager.end_exp(recorder_status)

    def search_records(self, experiment_ids, **kwargs):
        """
        Get a pandas DataFrame of records that fit the search criteria.

        The arguments of this function are not set to be rigid, and they will be different with different implementation of
        ``ExpManager`` in ``Qlib``. ``Qlib`` now provides an implementation of ``ExpManager`` with mlflow, and here is the
        example code of the this method with the ``MLflowExpManager``:

        .. code-block:: Python

            R.log_metrics(m=2.50, step=0)
            records = R.search_runs([experiment_id], order_by=["metrics.m DESC"])

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
        return self.exp_manager.search_records(experiment_ids, **kwargs)

    def list_experiments(self):
        """
        Method for listing all the existing experiments (except for those being deleted.)

        .. code-block:: Python

            exps = R.list_experiments()

        Returns
        -------
        A dictionary (name -> experiment) of experiments information that being stored.
        """
        return self.exp_manager.list_experiments()

    def list_recorders(self, experiment_id=None, experiment_name=None):
        """
        Method for listing all the recorders of experiment with given id or name.

        If user doesn't provide the id or name of the experiment, this method will try to retrieve the default experiment and
        list all the recorders of the default experiment. If the default experiment doesn't exist, the method will first
        create the default experiment, and then create a new recorder under it. (More information about the default experiment
        can be found `here <../component/recorder.html#qlib.workflow.exp.Experiment>`_).

        Here is the example code:

        .. code-block:: Python

            recorders = R.list_recorders(experiment_name='test')

        Parameters
        ----------
        experiment_id : str
            id of the experiment.
        experiment_name : str
            name of the experiment.

        Returns
        -------
        A dictionary (id -> recorder) of recorder information that being stored.
        """
        return self.get_exp(experiment_id, experiment_name).list_recorders()

    def get_exp(self, experiment_id=None, experiment_name=None, create: bool = True) -> Experiment:
        """
        Method for retrieving an experiment with given id or name. Once the `create` argument is set to
        True, if no valid experiment is found, this method will create one for you. Otherwise, it will
        only retrieve a specific experiment or raise an Error.

        - If '`create`' is True:

            - If `active experiment` exists:

                - no id or name specified, return the active experiment.

                - if id or name is specified, return the specified experiment. If no such exp found, create a new experiment with given id or name, and the experiment is set to be active.

            - If `active experiment` not exists:

                - no id or name specified, create a default experiment, and the experiment is set to be active.

                - if id or name is specified, return the specified experiment. If no such exp found, create a new experiment with given name or the default experiment, and the experiment is set to be active.

        - Else If '`create`' is False:

            - If ``active experiment` exists:

                - no id or name specified, return the active experiment.

                - if id or name is specified, return the specified experiment. If no such exp found, raise Error.

            - If `active experiment` not exists:

                - no id or name specified. If the default experiment exists, return it, otherwise, raise Error.

                - if id or name is specified, return the specified experiment. If no such exp found, raise Error.

        Here are some use cases:

        .. code-block:: Python

            # Case 1
            with R.start('test'):
                exp = R.get_exp()
                recorders = exp.list_recorders()

            # Case 2
            with R.start('test'):
                exp = R.get_exp('test1')

            # Case 3
            exp = R.get_exp() -> a default experiment.

            # Case 4
            exp = R.get_exp(experiment_name='test')

            # Case 5
            exp = R.get_exp(create=False) -> the default experiment if exists.

        Parameters
        ----------
        experiment_id : str
            id of the experiment.
        experiment_name : str
            name of the experiment.
        create : boolean
            an argument determines whether the method will automatically create a new experiment
            according to user's specification if the experiment hasn't been created before.

        Returns
        -------
        An experiment instance with given id or name.
        """
        return self.exp_manager.get_exp(experiment_id, experiment_name, create)

    def delete_exp(self, experiment_id=None, experiment_name=None):
        """
        Method for deleting the experiment with given id or name. At least one of id or name must be given,
        otherwise, error will occur.

        Here is the example code:

        .. code-block:: Python

            R.delete_exp(experiment_name='test')

        Parameters
        ----------
        experiment_id : str
            id of the experiment.
        experiment_name : str
            name of the experiment.
        """
        self.exp_manager.delete_exp(experiment_id, experiment_name)

    def get_uri(self):
        """
        Method for retrieving the uri of current experiment manager.

        Here is the example code:

        .. code-block:: Python

            uri = R.get_uri()

        Returns
        -------
        The uri of current experiment manager.
        """
        return self.exp_manager.get_uri()

    def get_recorder(self, recorder_id=None, recorder_name=None, experiment_name=None):
        """
        Method for retrieving a recorder.

        - If `active recorder` exists:

            - no id or name specified, return the active recorder.

            - if id or name is specified, return the specified recorder.

        - If `active recorder` not exists:

            - no id or name specified, raise Error.

            - if id or name is specified, and the corresponding experiment_name must be given, return the specified recorder. Otherwise, raise Error.

        The recorder can be used for further process such as `save_object`, `load_object`, `log_params`,
        `log_metrics`, etc.

        Here are some use cases:

        .. code-block:: Python

            # Case 1
            with R.start('test'):
                recorder = R.get_recorder()

            # Case 2
            with R.start('test'):
                recorder = R.get_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d')

            # Case 3
            recorder = R.get_recorder() -> Error

            # Case 4
            recorder = R.get_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d') -> Error

            # Case 5
            recorder = R.get_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d', experiment_name='test')

        Parameters
        ----------
        recorder_id : str
            id of the recorder.
        recorder_name : str
            name of the recorder.
        experiment_name : str
            name of the experiment.

        Returns
        -------
        A recorder instance.
        """
        return self.get_exp(experiment_name=experiment_name, create=False).get_recorder(
            recorder_id, recorder_name, create=False
        )

    def delete_recorder(self, recorder_id=None, recorder_name=None):
        """
        Method for deleting the recorders with given id or name. At least one of id or name must be given,
        otherwise, error will occur.

        Here is the example code:

        .. code-block:: Python

            R.delete_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d')

        Parameters
        ----------
        recorder_id : str
            id of the experiment.
        recorder_name : str
            name of the experiment.
        """
        self.get_exp().delete_recorder(recorder_id, recorder_name)

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        """
        Method for saving objects as artifacts in the experiment to the uri. It supports either saving
        from a local file/directory, or directly saving objects. User can use valid python's keywords arguments
        to specify the object to be saved as well as its name (name: value).

        - If `active recorder` exists: it will save the objects through the active recorder.
        - If `active recorder` not exists: the system will create a default experiment, and a new recorder and save objects under it.

        .. note::

            If one wants to save objects with a specific recorder. It is recommended to first get the specific recorder through `get_recorder` API and use the recorder the save objects. The supported arguments are the same as this method.

        Here are some use cases:

        .. code-block:: Python

            # Case 1
            with R.start('test'):
                pred = model.predict(dataset)
                R.save_objects(**{"pred.pkl": pred}, artifact_path='prediction')

            # Case 2
            with R.start('test'):
                R.save_objects(local_path='results/pred.pkl')

        Parameters
        ----------
        local_path : str
            if provided, them save the file or directory to the artifact URI.
        artifact_path : str
            the relative path for the artifact to be stored in the URI.
        """
        self.get_exp().get_recorder().save_objects(local_path, artifact_path, **kwargs)

    def log_params(self, **kwargs):
        """
        Method for logging parameters during an experiment. In addition to using ``R``, one can also log to a specific recorder after getting it with `get_recorder` API.

        - If `active recorder` exists: it will log parameters through the active recorder.
        - If `active recorder` not exists: the system will create a default experiment as well as a new recorder, and log parameters under it.

        Here are some use cases:

        .. code-block:: Python

            # Case 1
            with R.start('test'):
                R.log_params(learning_rate=0.01)

            # Case 2
            R.log_params(learning_rate=0.01)

        Parameters
        ----------
        keyword argument:
            name1=value1, name2=value2, ...
        """
        self.get_exp().get_recorder().log_params(**kwargs)

    def log_metrics(self, step=None, **kwargs):
        """
        Method for logging metrics during an experiment. In addition to using ``R``, one can also log to a specific recorder after getting it with `get_recorder` API.

        - If `active recorder` exists: it will log metrics through the active recorder.
        - If `active recorder` not exists: the system will create a default experiment as well as a new recorder, and log metrics under it.

        Here are some use cases:

        .. code-block:: Python

            # Case 1
            with R.start('test'):
                R.log_metrics(train_loss=0.33, step=1)

            # Case 2
            R.log_metrics(train_loss=0.33, step=1)

        Parameters
        ----------
        keyword argument:
            name1=value1, name2=value2, ...
        """
        self.get_exp().get_recorder().log_metrics(step, **kwargs)

    def set_tags(self, **kwargs):
        """
        Method for setting tags for a recorder. In addition to using ``R``, one can also set the tag to a specific recorder after getting it with `get_recorder` API.

        - If `active recorder` exists: it will set tags through the active recorder.
        - If `active recorder` not exists: the system will create a default experiment as well as a new recorder, and set the tags under it.

        Here are some use cases:

        .. code-block:: Python

            # Case 1
            with R.start('test'):
                R.set_tags(release_version="2.2.0")

            # Case 2
            R.set_tags(release_version="2.2.0")

        Parameters
        ----------
        keyword argument:
            name1=value1, name2=value2, ...
        """
        self.get_exp().get_recorder().set_tags(**kwargs)


import sys

if sys.version_info >= (3, 9):
    from typing import Annotated

    QlibRecorderWrapper = Annotated[QlibRecorder, Wrapper]
else:
    QlibRecorderWrapper = QlibRecorder

# global record
R: QlibRecorderWrapper = Wrapper()
