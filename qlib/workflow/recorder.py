# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
import shutil, os, pickle, tempfile, codecs
from pathlib import Path
from ..utils.objm import FileManager


class Recorder:
    """
    This is the `Recorder` class for logging the experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)

    The status of the recorder can be SCHEDULED, RUNNING, FINISHED, FAILED.
    """

    def __init__(self, name, experiment_id):
        self.id = None
        self.name = name
        self.experiment_id = experiment_id
        self.status = "SCHEDULED"
    
    def __repr__(self):
        return str(self.info)
    
    def __str__(self):
        return str(self.info)    

    @property
    def info(self):
        output = dict()
        output['class'] = "Recorder"
        output['id'] = self.id
        output['name'] = self.name
        output['experiment_id'] = self.experiment_id
        output['status'] = self.status

    def set_recorder_name(self, rname):
        self.recorder_name = rname

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        """
        Save objects such as prediction file or model checkpoints to the artifact URI.

        Parameters
        ----------
        data : any type
            the data to be saved.
        name : str
            name of the file to be saved.
        data_name_list : list
            list of (data, name) pairs
        local_path : str
            if provided, them save the file or directory to the artifact URI.
        artifact_path=None : str
            the relative path for the artifact to be stored in the URI.
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

    def start_run(self):
        """
        Start running or resuming the Recorder. The return value can be used as a context manager within a `with` block;
        otherwise, you must call end_run() to terminate the current run. (See `ActiveRun` class in mlflow)

        Parameters
        ----------

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

    def log_params(self, **kwargs):
        """
        Log a batch of params for the current run.

        Parameters
        ----------
        keyword arguments
            key, value pair to be logged as parameters.
        """
        raise NotImplementedError(f"Please implement the `log_params` method.")

    def log_metrics(self, step=None, **kwargs):
        """
        Log multiple metrics for the current run.

        Parameters
        ----------
        keyword arguments
            key, value pair to be logged as metrics.
        """
        raise NotImplementedError(f"Please implement the `log_metrics` method.")

    def set_tags(self, **kwargs):
        """
        Log a batch of tags for the current run.

        Parameters
        ----------
        keyword arguments
            key, value pair to be logged as tags.
        """
        raise NotImplementedError(f"Please implement the `set_tags` method.")

    def delete_tags(self, *keys):
        """
        Delete some tags from a run.

        Parameters
        ----------
        keys : series of strs of the keys
            all the name of the tag to be deleted.
        """
        raise NotImplementedError(f"Please implement the `delete_tags` method.")

    def list_artifacts(self, artifact_path=None):
        """
        Delete some tags from a run.

        Parameters
        ----------
        artifact_path=None : str
            the relative path for the artifact to be stored in the URI.

        Returns
        -------
        A list of artifacts information (name, path, etc.) that being stored.
        """
        raise NotImplementedError(f"Please implement the `list_artifacts` method.")


class MLflowRecorder(Recorder):
    """
    Use mlflow to implement a Recorder.

    Due to the fact that mlflow will only log artifact from a file or directory, we decide to
    use file manager to help maintain the objects in the project.
    """

    def __init__(self, name, experiment_id):
        super(MLflowRecorder, self).__init__(name, experiment_id)
        self.fm = None
        self.temp_dir = None

    def start_run(self):
        # start the run
        run = mlflow.start_run(self.id, self.experiment_id, self.name)
        # save the run id and artifact_uri
        self.id = run.info.run_id
        self.artifact_uri = run.info.artifact_uri
        self._uri = mlflow.get_tracking_uri()  # Fix!!! : this is not proper to have uri in recorder
        # set up file manager for saving objects
        self.temp_dir = tempfile.mkdtemp()
        self.fm = FileManager(Path(self.temp_dir).absolute())
        self.status = "RUNNING"
        return run

    def end_run(self, status):
        mlflow.end_run(status)
        self.status = status
        shutil.rmtree(self.temp_dir)

    def save_objects(self, data_name_list=None, local_path=None, artifact_path=None, **kwargs):
        client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        if local_path is not None:
            client.log_artifacts(self.id, local_path, artifact_path)
        elif kwargs.get('data') is not None and kwargs.get('name') is not None:
            data, name = kwargs.get('data'), kwargs.get('name')
            self.fm.save_obj(data, name)
            client.log_artifact(self.id, self.fm.path / name, artifact_path)
        elif kwargs.get('data_name_list') is not None:
            data_name_list = kwargs.get('data_name_list')
            self.fm.save_objs(data_name_list)
            client.log_artifacts(self.id, self.fm.path, artifact_path)
        else:
            raise Exception('Please provide valid arguments in order to save object properly.')

    def load_object(self, name):
        client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        path = client.download_artifacts(self.id, name)
        try:
            with Path(path).open("rb") as f:
                f.seek(0)
                return pickle.load(f)
        except:
            with codecs.open(path, mode="r", encoding="utf-8") as f:
                return f.read()

    def log_params(self, **kwargs):
        keys = list(kwargs.keys())
        if len(keys) == 0:
            mlflow.log_param(keys[0], kwargs.get(keys[0]))
        else:
            mlflow.log_params(dict(kwargs))

    def log_metrics(self, step=None, **kwargs):
        keys = list(kwargs.keys())
        if len(keys) == 0:
            mlflow.log_metric(keys[0], kwargs.get(keys[0]))
        else:
            mlflow.log_metrics(dict(kwargs))

    def set_tags(self, **kwargs):
        keys = list(kwargs.keys())
        if len(keys) == 0:
            mlflow.set_tag(keys[0], kwargs.get(keys[0]))
        else:
            mlflow.set_tags(dict(kwargs))

    def delete_tags(self, *keys):
        for count, key in enumerate(keys):
            mlflow.delete_tag(key)

    def get_artifact_uri(self, artifact_path=None):
        if self.artifact_uri is not None:
            return self.artifact_uri
        return mlflow.get_artifact_uri(artifact_path)

    def list_artifacts(self, artifact_path=None):
        client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        artifacts = client.list_artifacts(self.id, artifact_path)
        return artifacts
