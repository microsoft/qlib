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
    """

    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.recorder_id = None
        self.recorder_name = None

    def set_recorder_name(self, rname):
        self.recorder_name = rname

    def save_object(self, data=None, name=None, local_path=None, artifact_path=None):
        """
        Save object such as prediction file or model checkpoints to the artifact URI.

        Parameters
        ----------
        data : any type
            the data to be saved.
        name : str
            name of the file to be saved.
        local_path : str
            if provided, them save the file or directory to the artifact URI.
        artifact_path=None : str
            the relative path for the artifact to be stored in the URI.
        """
        raise NotImplementedError(f"Please implement the `save_object` method.")

    def save_objects(self, data_name_list=None, local_path=None, artifact_path=None):
        """
        Save objects such as prediction file or model checkpoints to the artifact URI.

        Parameters
        ----------
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

    def start_run(self, run_id=None, experiment_id=None, run_name=None, nested=False):
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
        raise NotImplementedError(f"Please implement the `log_tags` method.")

    def delete_tag(self, key):
        """
        Delete a tag from a run.

        Parameters
        ----------
        key : str
            the name of the tag to be deleted.
        """
        raise NotImplementedError(f"Please implement the `delete_tag` method.")


class MLflowRecorder(Recorder):
    """
    Use mlflow to implement a Recorder.

    Due to the fact that mlflow will only log artifact from a file or directory, we decide to
    use file manager to help maintain the objects in the project.
    """

    def __init__(self, experiment_id):
        super(MLflowRecorder, self).__init__(experiment_id)
        self.fm = None
        self.temp_dir = None

    def start_run(self, run_id=None, experiment_id=None, run_name=None, nested=False):
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
        self._uri = mlflow.get_tracking_uri()  # Fix!!! : this is not proper to have uri in recorder
        # set up file manager for saving objects
        self.temp_dir = tempfile.mkdtemp()
        self.fm = FileManager(Path(self.temp_dir).absolute())
        return run

    def end_run(self):
        mlflow.end_run()
        shutil.rmtree(self.temp_dir)

    def save_object(self, data=None, name=None, local_path=None, artifact_path=None):
        client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        if local_path is None:
            assert data is not None and name is not None, "Please provide data and name input."
            self.fm.save_obj(data, name)
            client.log_artifact(self.recorder_id, self.fm.path / name, artifact_path)
        else:
            assert local_path is not None, "Please provide a valid local path for the "
            client.log_artifact(self.recorder_id, local_path, artifact_path)

    def save_objects(self, data_name_list=None, local_path=None, artifact_path=None):
        client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        if local_path is None:
            assert data_name_list is not None, "Please provide data_name_list input."
            self.fm.save_objs(data_name_list)
            client.log_artifacts(self.recorder_id, self.fm.path, artifact_path)
        else:
            client.log_artifacts(self.recorder_id, local_path, artifact_path)

    def load_object(self, name):
        client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        path = client.download_artifacts(self.recorder_id, name)
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

    def delete_tag(self, key):
        mlflow.delete_tag(key)

    def get_artifact_uri(self, artifact_path=None):
        if self.artifact_uri is not None:
            return self.artifact_uri
        return mlflow.get_artifact_uri(artifact_path)

    def check(self, name, path=None):
        client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        artifacts = client.list_artifacts(self.recorder_id, path)
        for artifact in artifacts:
            if name in artifact.path:
                return True
        return False
