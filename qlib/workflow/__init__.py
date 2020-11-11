# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from .expm import MLflowExpManager
from ..utils import Wrapper
from ..config import C


class QlibRecorder:
    """
    A global system that helps to manage the experiments.
    """

    def __init__(self, exp_manager):
        self.exp_manager = exp_manager
        self.uri = C["exp_uri"]

    @contextmanager
    def start(self, experiment_name):
        run = self.start_exp(experiment_name)
        try:
            yield run
        except Exception as e:
            self.end_exp("FAILED")  # end the experiment if something went wrong
            raise e
        self.end_exp("FINISHED")

    def start_exp(self, experiment_name=None):
        return self.exp_manager.start_exp(experiment_name, self.uri)

    def end_exp(self, status):
        self.exp_manager.end_exp(status)

    def search_records(self, experiment_ids, **kwargs):
        return self.exp_manager.search_records(experiment_ids, **kwargs)

    def get_exp(self, experiment_id=None, experiment_name=None):
        return self.exp_manager.get_exp(experiment_id, experiment_name)

    def delete_exp(self, experiment_id):
        self.exp_manager.delete_exp(experiment_id)

    def get_uri(self):
        return self.exp_manager.get_uri()

    def get_recorder(self, recorder_id=None, recorder_name=None):
        return self.exp_manager.active_experiment.get_recorder(recorder_id, recorder_name)

    def save_objects(self, local_path=None, artifact_path=None, **kwargs):
        self.exp_manager.active_experiment.active_recorder.save_objects(local_path, artifact_path, **kwargs)

    def load_object(self, name):
        return self.exp_manager.active_experiment.active_recorder.load_object(name)

    def log_params(self, **kwargs):
        self.exp_manager.active_experiment.active_recorder.log_params(**kwargs)

    def log_metrics(self, step=None, **kwargs):
        self.exp_manager.active_experiment.active_recorder.log_metrics(step, **kwargs)

    def set_tags(self, **kwargs):
        self.exp_manager.active_experiment.active_recorder.set_tags(**kwargs)

    def delete_tag(self, *key):
        self.exp_manager.active_experiment.active_recorder.delete_tag(*key)


# global record
R = Wrapper()
