# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from typing import Optional
import mlflow
import shutil
import pickle
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

from qlib.utils.serial import Serializable
from qlib.utils.exceptions import LoadObjectError
from qlib.utils.paral import AsyncCaller

from ..log import TimeInspector, get_module_logger
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository

logger = get_module_logger("workflow")
# mlflow limits the length of log_param to 500, but this caused errors when using qrun, so we extended the mlflow limit.
mlflow.utils.validation.MAX_PARAM_VAL_LENGTH = 1000


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
        return "{name}(info={info})".format(name=self.__class__.__name__, info=self.info)

    def __str__(self):
        return str(self.info)

    def __hash__(self) -> int:
        return hash(self.info["id"])

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

        Please refer to the docs of qlib.workflow:R.save_objects

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

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file or directory as an artifact of the currently active run.

        Parameters
        ----------
        local_path : str
            Path to the file to write.
        artifact_path : Optional[str]
            If provided, the directory in ``artifact_uri`` to write to.
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

    def download_artifact(self, path: str, dst_path: Optional[str] = None) -> str:
        """
        Download an artifact file or directory from a run to a local directory if applicable,
        and return a local path for it.

        Parameters
        ----------
        path : str
            Relative source path to the desired artifact.
        dst_path : Optional[str]
            Absolute path of the local filesystem destination directory to which to
            download the specified artifacts. This directory must already exist.
            If unspecified, the artifacts will either be downloaded to a new
            uniquely-named directory on the local filesystem.

        Returns
        -------
        str
            Local path of desired artifact.
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

    Instead of using mlflow directly, we use another interface wrapping mlflow to log experiments.
    Though it takes extra efforts, but it brings users benefits due to following reasons.
    - It will be more convenient to change the experiment logging backend without changing any code in upper level
    - We can provide more convenience to automatically do some extra things and make interface easier. For examples:
        - Automatically logging the uncommitted code
        - Automatically logging part of environment variables
        - User can control several different runs by just creating different Recorder (in mlflow, you always have to switch artifact_uri and pass in run ids frequently)
    """

    def __init__(self, experiment_id, uri, name=None, mlflow_run=None):
        super(MLflowRecorder, self).__init__(experiment_id, name)
        self._uri = uri
        self._artifact_uri = None
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
            self._artifact_uri = mlflow_run.info.artifact_uri
        self.async_log = None

    def __repr__(self):
        name = self.__class__.__name__
        space_length = len(name) + 1
        return "{name}(info={info},\n{space}uri={uri},\n{space}artifact_uri={artifact_uri},\n{space}client={client})".format(
            name=name,
            space=" " * space_length,
            info=self.info,
            uri=self.uri,
            artifact_uri=self.artifact_uri,
            client=self.client,
        )

    def __hash__(self) -> int:
        return hash(self.info["id"])

    def __eq__(self, o: object) -> bool:
        if isinstance(o, MLflowRecorder):
            return self.info["id"] == o.info["id"]
        return False

    @property
    def uri(self):
        return self._uri

    @property
    def artifact_uri(self):
        return self._artifact_uri

    def get_local_dir(self):
        """
        This function will return the directory path of this recorder.
        """
        if self.artifact_uri is not None:
            local_dir_path = Path(self.artifact_uri.lstrip("file:")) / ".."
            local_dir_path = str(local_dir_path.resolve())
            if os.path.isdir(local_dir_path):
                return local_dir_path
            else:
                raise RuntimeError("This recorder is not saved in the local file system.")

        else:
            raise ValueError(
                "Please make sure the recorder has been created and started properly before getting artifact uri."
            )

    def start_run(self):
        # set the tracking uri
        mlflow.set_tracking_uri(self.uri)
        # start the run
        run = mlflow.start_run(self.id, self.experiment_id, self.name)
        # save the run id and artifact_uri
        self.id = run.info.run_id
        self._artifact_uri = run.info.artifact_uri
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status = Recorder.STATUS_R
        logger.info(f"Recorder {self.id} starts running under Experiment {self.experiment_id} ...")

        # NOTE: making logging async.
        # - This may cause delay when uploading results
        # - The logging time may not be accurate
        self.async_log = AsyncCaller()

        # TODO: currently, this is only supported in MLflowRecorder.
        # Maybe we can make this feature more general.
        self._log_uncommitted_code()

        self.log_params(**{"cmd-sys.argv": " ".join(sys.argv)})  # log the command to produce current experiment
        self.log_params(
            **{k: v for k, v in os.environ.items() if k.startswith("_QLIB_")}
        )  # Log necessary environment variables
        return run

    def _log_uncommitted_code(self):
        """
        Mlflow only log the commit id of the current repo. But usually, user will have a lot of uncommitted changes.
        So this tries to automatically to log them all.
        """
        # TODO: the sub-directories maybe git repos.
        # So it will be better if we can walk the sub-directories and log the uncommitted changes.
        for cmd, fname in [
            ("git diff", "code_diff.txt"),
            ("git status", "code_status.txt"),
            ("git diff --cached", "code_cached.txt"),
        ]:
            try:
                out = subprocess.check_output(cmd, shell=True)
                self.client.log_text(self.id, out.decode(), fname)  # this behaves same as above
            except subprocess.CalledProcessError:
                logger.info(f"Fail to log the uncommitted code of $CWD({os.getcwd()}) when run {cmd}.")

    def end_run(self, status: str = Recorder.STATUS_S):
        assert status in [
            Recorder.STATUS_S,
            Recorder.STATUS_R,
            Recorder.STATUS_FI,
            Recorder.STATUS_FA,
        ], f"The status type {status} is not supported."
        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.status != Recorder.STATUS_S:
            self.status = status
        if self.async_log is not None:
            # Waiting Queue should go before mlflow.end_run. Otherwise mlflow will raise error
            with TimeInspector.logt("waiting `async_log`"):
                self.async_log.wait()
        self.async_log = None
        mlflow.end_run(status)

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."
        if local_path is not None:
            path = Path(local_path)
            if path.is_dir():
                self.client.log_artifacts(self.id, local_path, artifact_path)
            else:
                self.client.log_artifact(self.id, local_path, artifact_path)
        else:
            temp_dir = Path(tempfile.mkdtemp()).resolve()
            for name, data in kwargs.items():
                path = temp_dir / name
                Serializable.general_dump(data, path)
                self.client.log_artifact(self.id, temp_dir / name, artifact_path)
            shutil.rmtree(temp_dir)

    def load_object(self, name, unpickler=pickle.Unpickler):
        """
        Load object such as prediction file or model checkpoint in mlflow.

        Args:
            name (str): the object name

            unpickler: Supporting using custom unpickler

        Raises:
            LoadObjectError: if raise some exceptions when load the object

        Returns:
            object: the saved object in mlflow.
        """
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."

        path = None
        try:
            path = self.client.download_artifacts(self.id, name)
            with Path(path).open("rb") as f:
                data = unpickler(f).load()
            return data
        except Exception as e:
            raise LoadObjectError(str(e)) from e
        finally:
            ar = self.client._tracking_client._get_artifact_repo(self.id)
            if isinstance(ar, AzureBlobArtifactRepository) and path is not None:
                # for saving disk space
                # For safety, only remove redundant file for specific ArtifactRepository
                shutil.rmtree(Path(path).absolute().parent)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_params(self, **kwargs):
        for name, data in kwargs.items():
            self.client.log_param(self.id, name, data)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_metrics(self, step=None, **kwargs):
        for name, data in kwargs.items():
            self.client.log_metric(self.id, name, data, step=step)

    def log_artifact(self, local_path, artifact_path: Optional[str] = None):
        self.client.log_artifact(self.id, local_path=local_path, artifact_path=artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
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
            raise ValueError(
                "Please make sure the recorder has been created and started properly before getting artifact uri."
            )

    def list_artifacts(self, artifact_path=None):
        assert self.uri is not None, "Please start the experiment and recorder first before using recorder directly."
        artifacts = self.client.list_artifacts(self.id, artifact_path)
        return [art.path for art in artifacts]

    def download_artifact(self, path: str, dst_path: Optional[str] = None) -> str:
        return self.client.download_artifacts(self.id, path, dst_path)

    def list_metrics(self):
        run = self.client.get_run(self.id)
        return run.data.metrics

    def list_params(self):
        run = self.client.get_run(self.id)
        return run.data.params

    def list_tags(self):
        run = self.client.get_run(self.id)
        return run.data.tags
