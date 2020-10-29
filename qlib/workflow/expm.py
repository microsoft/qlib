# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
import os
from pathlib import Path
from contextlib import contextmanager
from .exp import MLflowExperiment
from .record import MLflowRecorder

class ExpManager:
    """
    This is the `ExpManager` class for managing the experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """
    def __init__(self):
        self.default_uri = None
        self.active_recorder = None # only one recorder can running each time
        self.experiments = dict() # store the experiment name --> Experiment object

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
        An object wrapped by context manager.
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
        A pandas.DataFrame of runs.
        """
        raise NotImplementedError(f"Please implement the `search_records` method.")

    def __create_exp(self, experiment_name, artifact_location=None):
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

    def get_uri(self, type):
        """
        Get the default tracking URI or current URI.

        Parameters
        ----------
        type  : str
            the type of the tracking URI one wants to retrieve.

        Returns
        -------
        The tracking URI string.
        """
        raise NotImplementedError(f"Please implement the `create_exp` method.")

    def get_recorder(self):
        """
        Get the current active Recorder.

        Parameters
        ----------

        Returns
        -------
        An Recorder object.
        """
        raise NotImplementedError(f"Please implement the `get_recorder` method.")
        

class MLflowExpManager(ExpManager):
    '''
    Use mlflow to implement ExpManager.
    '''
    def __init__(self):
        super(MLflowExpManager, self).__init__()
        self.default_uri = None
        self.current_uri = None

    def start_exp(self, experiment_name=None, uri=None):
        # create experiment
        experiment = self.__create_exp(experiment_name, uri)
        # set up recorder
        recorder = MLflowRecorder(experiment.id)
        self.active_recorder = recorder
        # store the recorder
        experiment.recorders.append(self.active_recorder)
        # store the experiment
        self.experiments[experiment_name] = experiment

        return self.active_recorder.start_run(experiment_id=experiment.id)

    def end_exp(self):
        self.active_recorder.end_run()
        self.active_recorder = None
    
    def __create_exp(self, experiment_name=None, uri=None):
        # init experiment
        experiment = MLflowExperiment()
        # set the tracking uri
        if uri is None:
            print('No tracking URI is provided. The default tracking URI is set as `mlruns` under the working directory.')
        else:
            self.current_uri = uri
        mlflow.set_tracking_uri(self.current_uri)
        # start the experiment
        if experiment_name is None:
            print('No experiment name provided. The default experiment name is set as `experiment`.')
            experiment_id = mlflow.create_experiment('experiment')
            # set the active experiment
            mlflow.set_experiment('experiment')
            experiment_name = 'experiment'
        else:
            if experiment_name not in self.experiments:
                if mlflow.get_experiment_by_name(experiment_name) is not None:
                    raise Exception('The experiment has already been created before. Please pick another name or delete the files under uri.')
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
        filter_string = '' if kwargs.get('filter_string') is None else kwargs.get('filter_string')
        run_view_type = 1 if kwargs.get('run_view_type') is None else kwargs.get('run_view_type')
        max_results = 100000 if kwargs.get('max_results') is None else kwargs.get('max_results')
        order_by = kwargs.get('order_by')
        return mlflow.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by)
    
    def get_exp(self, experiment_id=None, experiment_name=None):
        assert experiment_id is not None or experiment_name is not None, 'Please provide at least one of the experiment id or name to retrieve an experiment.'
        if experiment_name is not None:
            return self.experiments[experiment_name]
        elif:
            for name in self.experiments:
                if self.experiments[name].id == experiment_id:
                    return self.experiments[name]
        else:
            print('No valid experiment is found. Please make sure the id and name are correctly given.')

    def delete_exp(self, experiment_id):
        mlflow.delete_experiment(experiment_id)
        self.experiments = {key:val for key, val in self.experiments.items() if val.id != experiment_id}

    def get_uri(self, type):
        if uri == 'default':
            return self.default_uri
        elif uri == 'current':
            return self.current_uri
        else:
            raise ValueError('Input type is not supported. Please choose type default or current to get the uri.')

    def get_recorder(self):
        return self.active_recorder