# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
from contextlib import contextmanager
from .record import MLflowRecorder

class ExpManager:
    """
    This is the `ExpManager` class for managing the experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """
    def __init__(self):
        self.active_recorder = None
        self.experiments = dict() # store the experiment names -> list of recorders.
        self.exp_ids = list()
    
    def _store_exp(self, id, name):
        """
        Store the experiments in the experiments holder.
        """
        if id in self.exp_ids:
            raise Exception('Something went wrong when creating the experiment. Please check if the experiment is already created.')
        if name in self.experiments:
            assert int(id) == int(self.experiments[name][0]), 'Experiment id and name are not consistent when storing the experiment.'
        else:
            self.exp_ids.append(id)
            self.experiments[name] = [id]

    def start_exp(self, project_path, experiment_name=None, uri=None, artifact_location=None, nested=False):
        """
        Start running an experiment. This method can only work in the `with` statement.

        Parameters
        ----------
        project_path : str
            path for the project.
        experiment_name : str
            name of the active experiment.
        uri : str
            the current tracking URI.
        artifact_location : str
            the location to store all the artifacts.
        nested : boolean
            controls whether run is nested in parent run.

        Returns
        None
        """
        raise NotImplementedError(f"Please implement the `start_exp` method.")
    
    def end_exp(self):
        """
        End an active experiment.
        """
        raise NotImplementedError(f"Please implement the `end_exp` method.")
    
    def search_runs(self, experiment_ids=None, filter_string='', run_view_type=1, max_results=100000, order_by=None):
        """
        Get a pandas DataFrame of runs that fit the search criteria.

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
        raise NotImplementedError(f"Please implement the `search_runs` method.")
    
    def get_exp(self, experiment_id):
        """
        Retrieve an experiment by experiment_id from the backend store.

        Parameters
        ----------
        experiment_id : str
            the experiment id to return.

        Returns
        -------
        An experiment object (e.g. mlflow.entities.Experiment).
        """
        raise NotImplementedError(f"Please implement the `get_exp` method.")

    def get_exp_by_name(self, experiment_name):
        """
        Retrieve an experiment by experiment name from the backend store.

        Parameters
        ----------
        experiment_name : str
            the experiment name to return.

        Returns
        -------
        An experiment object (e.g. mlflow.entities.Experiment).
        """
        raise NotImplementedError(f"Please implement the `get_exp_by_name` method.")
    
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
        String id of created experiment.
        """
        raise NotImplementedError(f"Please implement the `create_exp` method.")
    
    def set_exp(self, experiment_name):
        """
        Set the experiment to be active.

        Parameters
        ----------
        experiment_name : str
            the experiment name, which must be unique.

        Returns
        -------
        String id of created experiment.
        """
        raise NotImplementedError(f"Please implement the `set_exp` method.")

    def delete_exp(self, experiment_id):
        """
        Delete an experiment.

        Parameters
        ----------
        experiment_id  : str
            the experiment id.  

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `create_exp` method.")

    def set_tracking_uri(self, uri):
        """
        Set the tracking server URI.

        Parameters
        ----------
        uri : str
            the uri of the tracking server, can be An empty string, or a local file path, prefixed with file:/.
            or An HTTP URI or A Databricks workspace.
        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `set_tracking_uri` method.")
    
    def get_tracking_uri(self):
        """
        Get the tracking server URI.

        Parameters
        ----------

        Returns
        -------
        The tracking URI.
        """
        raise NotImplementedError(f"Please implement the `get_tracking_uri` method.")
    
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
    def start_exp(self, experiment_name=None, uri=None, project_path=None, artifact_location=None, nested=False):
        # set the tracking uri
        if uri is None:
            assert project_path is not None, "Please provide the project_path if no uri is provided in order to set a proper tracking uri."
            print('No tracking URI is provided. The default tracking URI is set as `mlruns` under the project path.')
            mlflow.set_tracking_uri(str(project_path / "mlruns"))
        else:
            mlflow.set_tracking_uri(uri)
        # start the experiment
        if experiment_name is None:
            print('No experiment name provided. The default experiment name is set as `experiment`.')
            experiment_id = self.create_exp('experiment', artifact_location)
            # set the active experiment
            self.set_exp('experiment')
            experiment_name = 'experiment'
        else:
            if experiment_name not in self.experiments:
                if self.get_exp_by_name(experiment_name) is not None:
                    raise Exception('The experiment has already been created before. Please pick another name or delete the files under tracking uri.')
                experiment_id = self.create_exp(experiment_name, artifact_location)
            else:
                experiment_id = self.experiments(experiment_name)[0]
            # set the active experiment
            self.set_exp(experiment_name)

        # store the id and name
        self._store_exp(experiment_id, experiment_name)
        # set up recorder
        recorder = MLflowRecorder(experiment_id)
        self.active_recorder = recorder
        # store the recorder
        self.experiments[experiment_name].append(self.active_recorder)

        return self.active_recorder.start_run(experiment_id=experiment_id, nested=nested)

    def search_runs(self, experiment_ids=None, filter_string='', run_view_type=1, max_results=100000, order_by=None):
        return mlflow.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by)
    
    def get_exp(self, experiment_id):
        return mlflow.get_experiment(experiment_id)

    def get_exp_by_name(self, experiment_name):
        return mlflow.get_experiment_by_name(experiment_name)

    def create_exp(self, experiment_name, artifact_location=None):
        return mlflow.create_experiment(experiment_name, artifact_location)

    def set_exp(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def delete_exp(self, experiment_id):
        mlflow.delete_experiment(experiment_id)
        self.experiments = {key:val for key, val in self.experiments.items() if val[0] != experiment_id}

    def set_tracking_uri(self, uri):
        mlflow.set_tracking_uri(uri)
    
    def get_tracking_uri(self):
        return mlflow.get_tracking_uri()
    
    def get_recorder(self):
        return self.active_recorder