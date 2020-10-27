# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
import shutil
from pathlib import Path
from ..utils.objm import FileManager

class Recorder:
    """
    This is the `Recorder` class for logging the experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)
    """

    def __init__(self, experiment_id, project_path=None):
        self.experiment_id = experiment_id
        self.recorder_id = None
        self.recorder_name = None
        self.fm = None
        self.artifact_uri = None
    
    def set_recorder_name(self, rname):
        self.recorder_name = rname

    def save_object(self, name, data):
        """
        Save object such as prediction file or model checkpoints.

        Parameters
        ----------
        name : str
            name of the file to be saved.
        data : any type
            the data to be saved.

        Returns
        -------
        None.
        """
        raise NotImplementedError(f"Please implement the `save_object` method.")

    def save_objects(self, name_data_list):
        """
        Save objects such as prediction file or model checkpoints.

        Parameters
        ----------
        name_data_list : list
            list of (name, data) pairs

        Returns
        -------
        None.
        """
        raise NotImplementedError(f"Please implement the `save_objects` method.")

    def load_object(self, name):
        """
        Load objects such as prediction file or model checkpoints.

        Parameters
        ----------
        name : str
            name of the file to be loaded.

        Returns
        -------
        The saved object.
        """
        raise NotImplementedError(f"Please implement the `load_object` method.")

    def start_run(self, run_id=None, experiment_id=None, 
                    run_name=None, nested=False):
        """
        Start running the Recorder. The return value can be used as a context manager within a `with` block; 
        otherwise, you must call end_run() to terminate the current run. (See `ActiveRun` class in mlflow)

        Parameters
        ----------
        run_id : str
            id of the active Recorder.
        experiment_id : str
            id of the active experiment.
        run_name : str
            name of the Recorder.
        nested : boolean
            controls whether run is nested in parent run.

        Returns
        -------
        An active running object (e.g. mlflow.ActiveRun object).
        """
        raise NotImplementedError(f"Please implement the `start_run` method.")
    
    def end_run(self):
        """
        End an active Recorder.
        """
        raise NotImplementedError(f"Please implement the `end_run` method.")

    def log_param(self, key, value):
        """
        Log a parameter under the current run.

        Parameters
        ----------
        key : str
            the name of the parameter
        value : str
            the value of the parameter

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `log_param` method.")

    def log_params(self, params):
        """
        Log a batch of params for the current run.

        Parameters
        ----------
        params : dict
            dictionary of param_name: String -> value: String.

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `log_params` method.")

    def log_metric(self, key, value, step=None):
        """
        Log a metric under the current run.

        Parameters
        ----------
        key : str
            the name of the metric
        value : float
            the value of the metric

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `log_metric` method.")

    def log_metrics(self, metrics, step=None):
        """
        Log multiple metrics for the current run.

        Parameters
        ----------
        metrics : dict
            dictionary of metric_name: String -> value: Float.

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `log_metrics` method.")
    
    def set_tag(self, key, value):
        """
        Set a tag under the current run.

        Parameters
        ----------
        key : str
            the name of the tag
        value : str
            the value of the tag

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `set_tag` method.")

    def set_tags(self, tags):
        """
        Log a batch of tags for the current run.

        Parameters
        ----------
        tags : dict
            dictionary of tag_name: String -> value: String.

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `log_tags` method.")

    def delete_tag(self, key):
        """
        Delete a tag from a run.

        Parameters
        ----------
        key : str
            the name of the tag to be deleted.

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `delete_tag` method.")
    
    def log_artifact(self, local_path, artifact_path=None):
        """
        Log a local file or directory as an artifact of the currently active run.

        Parameters
        ----------
        local_path : str
            path to the file to write.
        artifact_path : str
            the directory in `artifact_uri` to write to.

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `log_artifact` method.")

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log all the contents of a local directory as artifacts of the run.

        Parameters
        ----------
        local_dir : str
            path to the directory of files to write.
        artifact_path : str
            the directory in `artifact_uri` to write to.

        Returns
        -------
        None
        """
        raise NotImplementedError(f"Please implement the `log_artifacts` method.")

    def get_artifact_uri(self, artifact_path=None):
        """
        Get the absolute URI of the specified artifact in the currently active run.

        Parameters
        ----------
        artifact_path : str
            the directory in `artifact_uri` to write to.

        Returns
        -------
        An absolute URI referring to the specified artifact or currently active Recorder.
        """
        raise NotImplementedError(f"Please implement the `get_artifact_uri` method.")


class MLflowRecorder(Recorder):
    '''
    Use mlflow to implement a Recorder.
    '''
    def start_run(self, run_id=None, experiment_id=None, 
                    run_name=None, nested=False):
        if run_id is None:
            run_id = self.recorder_id
        if experiment_id is None:
            experiment_id = self.experiment_id
        if run_name is None:
            run_name = self.recorder_name
        # start the run
        run = mlflow.start_run(run_id, experiment_id, run_name, nested)
        # save the run id and artifact_uri
        self.recorder_id = run.info.run_id
        self.artifact_uri = run.info.artifact_uri
        # set up file manager for saving objects
        if self.artifact_uri.startswith('file:/'):
            self.fm = FileManager(Path(urllib.parse.urlparse(self.artifact_uri).path))
        else:
            self.fm = FileManager(Path(self.artifact_uri))
        print(self.artifact_uri)
        return run
    
    def end_run(self):
        mlflow.end_run()

    def save_object(self, name, data):
        self.fm.save_obj(data, name)
        import urllib
        print(urllib.parse.urlparse(self.artifact_uri).scheme)
        try:
            self.log_artifact(self.fm.path / name)
        except shutil.SameFileError:
            pass
        except Exception as e:
            print(e)

    def save_objects(self, name_data_list):
        self.fm.save_objs(name_data_list)
        try:
            self.log_artifacts(self.fm.path)
        except shutil.SameFileError:
            pass
        except Exception as e:
            print(e)

    def load_object(self, name):
        return self.fm.load_obj(name)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step)
    
    def set_tag(self, key, value):
        mlflow.set_tag(key, value)

    def set_tags(self, tags):
        mlflow.set_tags(tags)

    def delete_tag(self, key):
        mlflow.delete_tag(key)
    
    def log_artifact(self, local_path, artifact_path=None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        mlflow.log_artifacts(local_dir, artifact_path)

    def get_artifact_uri(self, artifact_path=None):
        if self.artifact_uri is not None:
            return self.artifact_uri
        return mlflow.get_artifact_uri(artifact_path)