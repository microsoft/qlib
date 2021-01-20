# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
import shutil, os, pickle, tempfile, codecs, pickle
from pathlib import Path
from datetime import datetime
from ..utils.objm import FileManager
from ..log import get_module_logger

logger = get_module_logger("workflow", "INFO")


class Recorder:
    """
    This is the `Recorder` class for logging the experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)

    The status of the recorder can be SCHEDULED, RUNNING, FINISHED, FAILED.
    """

    # status type
    STATUS_S = "SCHEDULED"
    STATUS_R = "RUNNING"
    STATUS_FI = "FINISHED"
    STATUS_FA = "FAILED"

    def __init__(self, experiment_id, name):
        self.id = None
        self.name = name
        self.experiment_id = experiment_id
        self.start_time = None
        self.end_time = None
        self.status = Recorder.STATUS_S

    def __repr__(self):
        return str(self.info)

    def __str__(self):
        return str(self.info)

    @property
    def info(self):
        output = dict()
        output["class"] = "Recorder"
        output["id"] = self.id
        output["name"] = self.name
        output["experiment_id"] = self.experiment_id
        output["start_time"] = self.start_time
        output["end_time"] = self.end_time
        output["status"] = self.status
        return output

    def set_recorder_name(self, rname):
        self.recorder_name = rname

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        """
        Save objects such as prediction file or model checkpoints to the artifact URI. User
        can save object through keywords arguments (name:value).

        Parameters
        ----------
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

    def list_artifacts(self, artifact_path: str = None):
        """
        List all the artifacts of a recorder.

        Parameters
        ----------
        artifact_path : str
            the relative path for the artifact to be stored in the URI.

        Returns
        -------
        A list of artifacts information (name, path, etc.) that being stored.
        """
        raise NotImplementedError(f"Please implement the `list_artifacts` method.")

    def list_metrics(self):
        """
        List all the metrics of a recorder.

        Returns
        -------
        A dictionary of metrics that being stored.
        """
        raise NotImplementedError(f"Please implement the `list_metrics` method.")

    def list_params(self):
        """
        List all the params of a recorder.

        Returns
        -------
        A dictionary of params that being stored.
        """
        raise NotImplementedError(f"Please implement the `list_params` method.")

    def list_tags(self):
        """
        List all the tags of a recorder.

        Returns
        -------
        A dictionary of tags that being stored.
        """
        raise NotImplementedError(f"Please implement the `list_tags` method.")


class MLflowRecorder(Recorder):
    """
    Use mlflow to implement a Recorder.

    Due to the fact that mlflow will only log artifact from a file or directory, we decide to
    use file manager to help maintain the objects in the project.
    """

    def __init__(self, experiment_id, uri, name=None, mlflow_run=None):
        super(MLflowRecorder, self).__init__(experiment_id, name)
        self._uri = uri
        self.artifact_uri = None
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self._uri)
        # construct from mlflow run
        if mlflow_run is not None:
            assert isinstance(mlflow_run, mlflow.entities.run.Run), "Please input with a MLflow Run object."
            self.name = mlflow_run.data.tags["mlflow.runName"]
            self.id = mlflow_run.info.run_id
            self.status = mlflow_run.info.status
            self.start_time = (
                datetime.fromtimestamp(float(mlflow_run.info.start_time) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                if mlflow_run.info.start_time is not None
                else None
            )
            self.end_time = (
                datetime.fromtimestamp(float(mlflow_run.info.end_time) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                if mlflow_run.info.end_time is not None
                else None
            )

    def start_run(self):
        # set the tracking uri
        mlflow.set_tracking_uri(self._uri)
        # start the run
        run = mlflow.start_run(self.id, self.experiment_id, self.name)
        # save the run id and artifact_uri
        self.id = run.info.run_id
        self.artifact_uri = run.info.artifact_uri
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status = Recorder.STATUS_R
        logger.info(f"Recorder {self.id} starts running under Experiment {self.experiment_id} ...")

        return run

    def end_run(self, status: str = Recorder.STATUS_S):
        assert status in [
            Recorder.STATUS_S,
            Recorder.STATUS_R,
            Recorder.STATUS_FI,
            Recorder.STATUS_FA,
        ], f"The status type {status} is not supported."
        mlflow.end_run(status)
        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.status != Recorder.STATUS_S:
            self.status = status

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        assert self._uri is not None, "Please start the experiment and recorder first before using recorder directly."
        if local_path is not None:
            self.client.log_artifacts(self.id, local_path, artifact_path)
        else:
            temp_dir = Path(tempfile.mkdtemp()).resolve()
            for name, data in kwargs.items():
                with (temp_dir / name).open("wb") as f:
                    pickle.dump(data, f)
                self.client.log_artifact(self.id, temp_dir / name, artifact_path)
            shutil.rmtree(temp_dir)

    def load_object(self, name):
        assert self._uri is not None, "Please start the experiment and recorder first before using recorder directly."
        path = self.client.download_artifacts(self.id, name)
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def log_params(self, **kwargs):
        for name, data in kwargs.items():
            self.client.log_param(self.id, name, data)

    def log_metrics(self, step=None, **kwargs):
        for name, data in kwargs.items():
            self.client.log_metric(self.id, name, data, step=step)

    def set_tags(self, **kwargs):
        for name, data in kwargs.items():
            self.client.set_tag(self.id, name, data)

    def delete_tags(self, *keys):
        for key in keys:
            self.client.delete_tag(self.id, key)

    def get_artifact_uri(self):
        if self.artifact_uri is not None:
            return self.artifact_uri
        else:
            raise Exception(
                "Please make sure the recorder has been created and started properly before getting artifact uri."
            )

    def list_artifacts(self, artifact_path=None):
        assert self._uri is not None, "Please start the experiment and recorder first before using recorder directly."
        artifacts = self.client.list_artifacts(self.id, artifact_path)
        return [art.path for art in artifacts]

    def list_metrics(self):
        run = self.client.get_run(self.id)
        return run.data.metrics

    def list_params(self):
        run = self.client.get_run(self.id)
        return run.data.params

    def list_tags(self):
        run = self.client.get_run(self.id)
        return run.data.tags
