# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from .expm import MLflowExpManager
from .recorder import Recorder
from ..utils import Wrapper


class QlibRecorder:
    """
    A global system that helps to manage the experiments.

    The components of the system:
    1) ExperimentManager: a class managing experiments.
    2) Experiment: a class of experiment, and each instance of it is responsible for a single experiment.
    3) Recorder: a class of recorder, and each instance of it is responsible for a single run.

    The general structure of the system:
    ExperimentManager
        - Experiment 1
            - Recorder 1
            - Recorder 2
            - ...
        - Experiment 2
            - ...
        - ...

    """

    def __init__(self, exp_manager):
        self.exp_manager = exp_manager

    @contextmanager
    def start(self, experiment_name=None):
        """
        Method to start an experiment. This method can only be called within a Python's `with` statement.

        Use case:
        ---------
        ```
        with R.start('test'):
            model.fit(dataset)
            R.log...
            ... # further operations
        ```

        Parameters
        ----------
        experiment_name : str
            name of the experiment one wants to start.
        """
        run = self.start_exp(experiment_name)
        try:
            yield run
        except Exception as e:
            self.end_exp(Recorder.STATUS_FA)  # end the experiment if something went wrong
            raise e
        self.end_exp(Recorder.STATUS_FI)

    def start_exp(self, experiment_name=None, uri=None):
        """
        Lower level method for starting an experiment. When use this method, one should end the experiment manually
        and the status of the recorder may not be handled properly.

        Use case:
        ---------
        ```
        R.start_exp(experiment_name='test')
        ... # further operations
        R.end_exp('FINISHED') or R.end_exp(Recorder.STATUS_S)
        ```

        Parameters
        ----------
        experiment_name : str
            the name of the experiment to be started
        uri : str
            the tracking uri of the experiment, where all the artifacts/metrics etc. will be stored.

        Returns
        -------
        An experiment instance being started.
        """
        return self.exp_manager.start_exp(experiment_name, uri)

    def end_exp(self, recorder_status=Recorder.STATUS_FI):
        """
        Method for ending an experiment manually. It will end the current active experiment, as well as its
        active recorder with the specified `status` type.

        Use case:
        ---------
        ```
        R.start_exp(experiment_name='test')
        ... # further operations
        R.end_exp('FINISHED') or R.end_exp(Recorder.STATUS_S)
        ```

        Parameters
        ----------
        status : str
            The status of a recorder, which can be SCHEDULED, RUNNING, FINISHED, FAILED.
        """
        self.exp_manager.end_exp(recorder_status)

    def search_records(self, experiment_ids, **kwargs):
        """
        Get a pandas DataFrame of records that fit the search criteria.

        Use case:
        ---------
        ```
        R.log_metrics(m=2.50, step=0)
        records = R.search_runs([experiment_id], order_by=["metrics.m DESC"])
        ```

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

        Use case:
        ---------
        ```
        exps = R.list_experiments()
        ```

        Returns
        -------
        A dictionary (name -> experiment) of experiments information that being stored.
        """
        return self.exp_manager.list_experiments()

    def list_recorders(self, experiment_id=None, experiment_name=None):
        """
        Method for listing all the recorders of experiment with given id or name.

        Use case:
        ---------
        ```
        recorders = R.list_recorders(experiment_name='test')
        ```

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

    def get_exp(self, experiment_id=None, experiment_name=None, create: bool = True):
        """
        Method for retrieving an experiment with given id or name. Once the `create` argument is set to
        True, if no valid experiment is found, this method will create one for you. Otherwise, it will
        only retrieve a specific experiment or raise an Error.

        If `create` is True:
            If R's running:
                1) no id or name specified, return the active experiment.
                2) if id or name is specified, return the specified experiment. If no such exp found,
                create a new experiment with given id or name, and the experiment is set to be running.
            If R's not running:
                1) no id or name specified, create a default experiment.
                2) if id or name is specified, return the specified experiment. If no such exp found,
                create a new experiment with given id or name, and the experiment is set to be running.
        Else If `create` is False:
            If R's running:
                1) no id or name specified, return the active experiment.
                2) if id or name is specified, return the specified experiment. If no such exp found,
                raise Error.
            If R's not running:
                1) no id or name specified. If the default experiment exists, return it, otherwise, raise Error.
                2) if id or name is specified, return the specified experiment. If no such exp found,
                raise Error.

        Use case:
        ---------
        ```
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
        ```

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

        Use case:
        ---------
        ```
        R.delete_exp(experiment_name='test')
        ```

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

        Use case:
        ---------
        ```
        uri = R.get_uri()
        ```

        Returns
        -------
        The uri of current experiment manager.
        """
        return self.exp_manager.get_uri()

    def get_recorder(self, recorder_id=None, recorder_name=None, experiment_name=None):
        """
        Method for retrieving a recorder.

        If R's running: 1) no id or name specified, return the active recorder. 2) if id or name is
        specified, return the specified recorder.
        If R's not running: 1) no id or name specified, raise Error. 2) if id or name is specified,
        and the corresponding experiment_name must be given, return the specified recorder. Otherwise,
        raise Error.

        The recorder can be used for further process such as `save_object`, `load_object`, `log_params`,
        `log_metrics`, etc.

        Use case:
        ---------
        ```
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
        ```

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

        Use case:
        ---------
        ```
        R.delete_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d')
        ```

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

        If R's running: it will save the objects through the running recorder.
        If R's not running: the system will create a default experiment, and a new recorder and
        save objects under it.

        If one wants to save objects with a specific recorder. It is recommended to first
        get the specific recorder through `get_recorder` API and use the recorder the save objects.
        The supported arguments are the same as this method.

        Use case:
        ---------
        ```
        # Case 1
        with R.start('test'):
            pred = model.predict(dataset)
            kwargs = {"pred.pkl": pred}
            R.save_objects(**kwargs, artifact_path='prediction')

        # Case 2
        with R.start('test'):
            R.save_objects(local_path='results/pred.pkl')
        ```

        Parameters
        ----------
        local_path : str
            if provided, them save the file or directory to the artifact URI.
        artifact_path=None : str
            the relative path for the artifact to be stored in the URI.
        """
        self.get_exp().get_recorder().save_objects(local_path, artifact_path, **kwargs)

    def log_params(self, **kwargs):
        """
        Method for logging parameters during an experiment.

        If R's running: it will log parameters through the running recorder.
        If R's not running: the system will create a default experiment as well as a new recorder, and
        log parameters under it.

        One can also log to a specific recorder after getting it with `get_recorder` API.

        Use case:
        ---------
        ```
        # Case 1
        with R.start('test'):
            R.log_params(learning_rate=0.01)

        # Case 2
        R.log_params(learning_rate=0.01)
        ```

        Parameters
        ----------
        keyword argument:
            name1=value1, name2=value2, ...
        """
        self.get_exp().get_recorder().log_params(**kwargs)

    def log_metrics(self, step=None, **kwargs):
        """
        Method for logging metrics during an experiment.

        If R's running: it will log metrics through the running recorder.
        If R's not running: the system will create a default experiment as well as a new recorder, and
        log metrics under it.

        One can also log to a specific recorder after getting it with `get_recorder` API.

        Use case:
        ---------
        ```
        # Case 1
        with R.start('test'):
            R.log_metrics(train_loss=0.33, step=1)

        # Case 2
        R.log_metrics(train_loss=0.33, step=1)
        ```

        Parameters
        ----------
        keyword argument:
            name1=value1, name2=value2, ...
        """
        self.get_exp().get_recorder().log_metrics(step, **kwargs)

    def set_tags(self, **kwargs):
        """
        Method for setting tags for a recorder.

        If R's running: it will set tags through the running recorder.
        If R's not running: the system will create a default experiment as well as a new recorder, and
        set the tags under it.

        One can also set the tag to a specific recorder after getting it with `get_recorder` API.

        Use case:
        ---------
        ```
        # Case 1
        with R.start('test'):
            R.set_tags(release_version="2.2.0")

        # Case 2
        R.set_tags(release_version="2.2.0")
        ```

        Parameters
        ----------
        keyword argument:
            name1=value1, name2=value2, ...
        """
        self.get_exp().get_recorder().set_tags(**kwargs)


# global record
R = Wrapper()
