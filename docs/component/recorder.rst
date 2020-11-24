.. _recorder:

====================================
Qlib Recorder: Experiment Management
====================================
.. currentmodule:: qlib

Introduction
===================
``Qlib`` contains an experiment management system named ``QlibRecorder``, which is designed to help users handle experiment and analysis results in an efficient way. 

There are three components of the system:

- `ExperimentManager`
    a class that manages experiments.

- `Experiment`
    a class of experiment, and each instance of it is responsible for a single experiment.

- `Recorder`
    a class of recorder, and each instance of it is responsible for a single run.

Here is a general view of the structure of the system:

.. code-block::

    ExperimentManager
        - Experiment 1
            - Recorder 1
            - Recorder 2
            - ...
        - Experiment 2
            - Recorder 1
            - Recorder 2
            - ...
        - ...

Currently, the components of this experiment management system are implemented using the machine learning platform: ``MLFlow`` (`link <https://mlflow.org/>`_). 


Qlib Recorder
===================
``QlibRecorder`` provides a high level API for users to use the experiment management system. The interfaces are wrapped in the variable ``R`` in ``Qlib``, and users can directly use ``R`` to interact with the system. The following command shows how to import ``R`` in Python:

.. code-block:: Python

        from qlib.workflow import R

``QlibRecorder`` includes several common API for managing `experiments` and `recorders` within a workflow. For more available APIs, please refer to the following section about `Experiment Manager`, `Experiment` and `Recorder`.

Here are the available interfaces of ``QlibRecorder``:

- `__init__(exp_manager)`
    - Initialization.
    - It takes in an input: `exp_manager`, which is an `ExperimentManager` instance. The instance will be created during ``qlib.init``.

- `start(experiment_name=None, recorder_name=None)`
    - High level API to start an experiment. This method can only be called within a Python's '`with`' statement.
    - Parameters:
        - `experiment_name` : str
            name of the experiment one wants to start.
        - `recorder_name` : str
            name of the recorder under the experiment one wants to start.
    - Use case:

    .. code-block:: Python

        with R.start('test', 'recorder_1'):
            model.fit(dataset)
            R.log...
            ... # further operations

- `start_exp(experiment_name=None, recorder_name=None, uri=None)`
    - Lower level method for starting an experiment. When use this method, one should end the experiment manually and the status of the recorder may not be handled properly.
    - Parameters:
        - `experiment_name` : str
            the name of the experiment to be started
        - `recorder_name` : str
            name of the recorder under the experiment one wants to start.
        - `uri` : str
            the tracking uri of the experiment, where all the artifacts/metrics etc. will be stored.
            The default uri are set in the qlib.config.
    - Returns: 
        - an experiment instance being started.
    - Use case:
    
    .. code-block:: Python

        R.start_exp(experiment_name='test', recorder_name='recorder_1')
        ... # further operations
        R.end_exp('FINISHED') or R.end_exp(Recorder.STATUS_S)

- `end_exp(recorder_status=Recorder.STATUS_FI)`
    - Method for ending an experiment manually. It will end the current active experiment, as well as its active recorder with the specified `status` type.
    - Parameters:
        - `status` : str
            The status of a recorder, which can be '`SCHEDULED`', '`RUNNING`', '`FINISHED`', '`FAILED`'.
    - Use case:

    .. code-block:: Python

        R.start_exp(experiment_name='test')
        ... # further operations
        R.end_exp('FINISHED') or R.end_exp(Recorder.STATUS_S)

- `search_records(experiment_ids, **kwargs)`
    - Get a pandas DataFrame of all the records that have been stored with the given search criteria. This method is highly correlated with MLFlow's ``search_runs`` method (`link <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs>`_).
    - Parameters:
        - `experiment_ids` : list
            list of experiment IDs.
        - `filter_string` : str
            filter query string, defaults to searching all runs.
        - `run_view_type` : int
            one of enum values ACTIVE_ONLY (1), DELETED_ONLY (2), or ALL (3).
        - `max_results`  : int
            the maximum number of runs to put in the dataframe.
        - `order_by` : list
            list of columns to order by (e.g., “metrics.rmse”).
    - Returns:
        -  A pandas.DataFrame of records, where each metric, parameter, and tag are expanded into their own columns named metrics.*, params.*, and tags.* respectively. For records that don't have a particular metric, parameter, or tag, their value will be (NumPy) Nan, None, or None respectively.
    - Use case:

    .. code-block:: Python

        R.log_metrics(m=2.50, step=0)
        records = R.search_runs([experiment_id], order_by=["metrics.m DESC"])

- `list_experiments()`
    - Method for listing all the existing experiments (except for those being deleted.)
    - Returns:
        - A dictionary (name -> experiment) of experiments information that being stored.
    - Use case:

    .. code-block:: Python

        exps = R.list_experiments()

- `list_recorders(experiment_id=None, experiment_name=None)`
    - Method for listing all the recorders of experiment with given id or name. If user doesn't provide the id or name of the experiment, this method will try to retrieve the default experiment and list all the recorders of the default experiment. If the default experiment doesn't exist, the method will first create the default experiment, and then create a new recorder under it. 
    - Parameters:
        - `experiment_id` : str
            id of the experiment.
        - `experiment_name` : str
            name of the experiment.
    - Returns:
        - A dictionary (id -> recorder) of recorder information that being stored.
    - Use case:

    .. code-block:: Python

        recorders = R.list_recorders(experiment_name='test')

- `get_exp(experiment_id=None, experiment_name=None, create: bool = True)`
    - Method for retrieving an experiment with given id or name. Once the '`create`' argument is set to True, if no valid experiment is found, this method will create one for the user. Otherwise, it will only retrieve a specific experiment or raise an Error.
        
        - If '`create`' is True:
            - If ``R``'s running:
                - no id or name specified, return the active experiment.
                - if id or name is specified, return the specified experiment. If no such exp found, create a new experiment with given id or name, and the experiment is set to be running.
            - If ``R``'s not running:
                - no id or name specified, create a default experiment, and the experiment is set to be running.
                - if id or name is specified, return the specified experiment. If no such exp found, create a new experiment with given name or the default experiment, and the experiment is set to be running.
        - Else If '`create`' is False:
            - If ``R``'s running:
                - no id or name specified, return the active experiment.
                - if id or name is specified, return the specified experiment. If no such exp found, raise Error.
            - If ``R``'s not running:
                - no id or name specified. If the default experiment exists, return it, otherwise, raise Error.
                - if id or name is specified, return the specified experiment. If no such exp found, raise Error.
    - Parameters:
        - `experiment_id` : str
            id of the experiment.
        - `experiment_name` : str
            name of the experiment.
        - `create` : boolean
            an argument determines whether the method will automatically create a new experiment according to user's specification if the experiment hasn't been created before.
    - Returns:
        - An experiment instance with given id or name.
    - Use case:

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

- `delete_exp(experiment_id=None, experiment_name=None)`
    - Method for deleting the experiment with given id or name. At least one of id or name must be given, otherwise, error will occur.
    - Parameters:
        - `experiment_id` : str
            id of the experiment.
        - `experiment_name` : str
            name of the experiment.
    - Use case:
    
    .. code-block:: Python

        R.delete_exp(experiment_name='test')

- `get_uri()`
    - Method for retrieving the uri of current experiment manager.
    - Returns:
        - The uri of current experiment manager.
    - Use case:
    
    .. code-block:: Python

        uri = R.get_uri()

- `get_recorder(recorder_id=None, recorder_name=None, experiment_name=None)`
    - Method for retrieving a recorder. The recorder can be used for further process such as ``save_objects``, ``load_object``, ``log_params``, ``log_metrics``, etc.

        - If ``R``'s running:
            - no id or name specified, return the active recorder.
            - if id or name is specified, return the specified recorder.
        - If ``R``'s not running:
            - no id or name specified, raise Error.
            - if id or name is specified, and the corresponding experiment_name must be given, return the specified recorder. Otherwise, raise Error.
    - Parameters:
        - `recorder_id` : str
            id of the recorder.
        - `recorder_name` : str
            name of the recorder.
        - `experiment_name` : str
            name of the experiment.
    - Returns:
        - A recorder instance.
    - Use case:
    
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

- `delete_recorder(recorder_id=None, recorder_name=None)`
    - Method for deleting the recorders with given id or name. At least one of id or name must be given, otherwise, error will occur.
    - Parameters:
        - `recorder_id` : str
            id of the experiment.
        - `recorder_name` : str
            name of the experiment.
    - Use case:
    
    .. code-block:: Python

        R.delete_recorder(recorder_id='2e7a4efd66574fa49039e00ffaefa99d')

- `save_objects(local_path=None, artifact_path=None, **kwargs)`
    - Method for saving objects as artifacts in the experiment to the uri. It supports either saving from a local file/directory, or directly saving objects. User can use valid python's keywords arguments to specify the object to be saved as well as its name (name: value).
        
        - If R's running: it will save the objects through the running recorder.
        - If R's not running: the system will create a default experiment, and a new recorder and save objects under it.
    
    .. note:: 

        If one wants to save objects with a specific recorder. It is recommended to first get the specific recorder through `get_recorder` API and use the recorder the save objects. The supported arguments are the same as this method.

    - Parameters:
        - `local_path` : str
            if provided, them save the file or directory to the artifact URI.
        - `artifact_path` : str
            the relative path for the artifact to be stored in the URI.
    - Use case:

    .. code-block:: Python

        # Case 1
        with R.start('test'):
            pred = model.predict(dataset)
            R.save_objects(**{"pred.pkl": pred}, artifact_path='prediction')

        # Case 2
        with R.start('test'):
            R.save_objects(local_path='results/pred.pkl')

- `log_params(**kwargs)`
    - Method for logging parameters during an experiment. In addition to using ``R``, one can also log to a specific recorder after getting it with `get_recorder` API.
    
        - If R's running: it will log parameters through the running recorder.
        - If R's not running: the system will create a default experiment as well as a new recorder, and log parameters under it.
    - Parameters:
        - `keyword argument`:
            name1=value1, name2=value2, ...
    - Use case:

    .. code-block:: Python

        # Case 1
        with R.start('test'):
            R.log_params(learning_rate=0.01)

        # Case 2
        R.log_params(learning_rate=0.01)

- `log_metrics(step=None, **kwargs)`
    - Method for logging metrics during an experiment. In addition to using ``R``, one can also log to a specific recorder after getting it with `get_recorder` API.

        - If R's running: it will log metrics through the running recorder.
        - If R's not running: the system will create a default experiment as well as a new recorder, and log metrics under it.
    - Parameters:
        - `step`: int
            a single integer step at which to log the specified Metrics. If unspecified, each metric is logged at step zero.
        - `keyword argument`:
            name1=value1, name2=value2, ...

- `set_tags(**kwargs)`
    - Method for setting tags for a recorder. In addition to using ``R``, one can also set the tag to a specific recorder after getting it with `get_recorder` API.
    
        - If R's running: it will set tags through the running recorder.
        - If R's not running: the system will create a default experiment as well as a new recorder, and set the tags under it.
    - Parameters:
        - `keyword argument`:
            name1=value1, name2=value2, ...
    - Use case:

    .. code-block:: Python

        # Case 1
        with R.start('test'):
            R.set_tags(release_version="2.2.0")
        
        # Case 2
        R.set_tags(release_version="2.2.0")


Experiment Manager
===================

The ``ExpManager`` module in ``Qlib`` is responsible for managing different experiments. Most of the APIs of ``ExpManager`` are similar to ``QlibRecorder``, and the most important API will be the ``get_exp`` method. User can directly refer to the documents above for some detailed information about how to use the ``get_exp`` method.

For other interfaces such as `create_exp`, `delete_exp`, please refer to `Experiment Manager API <../reference/api.html#experiment-manager>`_.

Experiment
===================

The ``Experiment`` class is solely responsible for a single experiment, and it will handle any operations that are related to an experiment. Basic methods such as `start`, `end` an experiment are included. Besides, methods related to `recorders` are also available: such methods include `get_recorder` and `list_recorders`.

For other interfaces such as `search_records`, `delete_recorder`, please refer to `Experiment API <../reference/api.html#experiment>`_.

Recorder
===================

The ``Recorder`` class is responsible for a single recorder. It will handle some detailed operations such as ``log_metrics``, ``log_params`` of a single run. It is designed to help user to easily track results and things being generated during a run.

Here are some important APIs that are not included in the ``QlibRecorder``:

- `list_artifacts(artifact_path: str = None)`
    - List all the artifacts of a recorder.
    - Parameters:
        - `artifact_path` : str
            the relative path for the artifact to be stored in the URI.
    - Returns:
        - A list of artifacts information (name, path, etc.) that being stored.

- `list_metrics()`
    - List all the metrics of a recorder.
    - Returns:
        - A dictionary of metrics that being stored.

- `list_params()`
    - List all the params of a recorder.
    - Returns:
        - A dictionary of params that being stored.

- `list_tags()`
    - List all the tags of a recorder.
    - Returns:
        - A dictionary of tags that being stored. 

For other interfaces such as `save_objects`, `load_object`, please refer to `Recorder API <../reference/api.html#recorder>`_.

Record Template
===================

The ``RecordTemp`` class is a class that enables generate experiment results such as IC and backtest in a certain format. We have provided three different `Record Template` class:

- ``SignalRecord``: This class generates the `preidction` of the model.
- ``SigAnaRecord``: This class generates the `IC`, `ICIR`, `Rank IC` and `Rank ICIR`.
- ``PortAnaRecord``: This class generates the results of `backtest`. The detailed information about `backtest` as well as the available `strategy`, users can refer to `Strategy <../component/strategy.html>`_ and `Backtest <../component/backtest.html>`_.

For more information, please refer to `Record Template API <../reference/api.html#module-qlib.workflow.record_temp>`_.