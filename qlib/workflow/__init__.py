# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from .record import MLflowRecorder
from .exp import MLflowExpManager

class Record:
    def __init__(self):
        pass

    @contextmanager
    def start_exp(self, experiment_name=None, uri=None, project_path=None, artifact_location=None, nested=False):
        raise NotImplementedError(f"Please implement the `start_exp` method.")
    
    def search_runs(self, experiment_ids=None, filter_string='', run_view_type=1, max_results=100000, order_by=None):
        raise NotImplementedError(f"Please implement the `search_runs` method.")
    
    def get_exp(self, experiment_id):
        raise NotImplementedError(f"Please implement the `get_exp` method.")

    def get_exp_by_name(self, experiment_name):
        raise NotImplementedError(f"Please implement the `get_exp_by_name` method.")
    
    def create_exp(self, experiment_name, artifact_location=None):
        raise NotImplementedError(f"Please implement the `create_exp` method.")
    
    def set_exp(self, experiment_name):
        raise NotImplementedError(f"Please implement the `set_exp` method.")

    def delete_exp(self, experiment_id):
        raise NotImplementedError(f"Please implement the `create_exp` method.")

    def set_tracking_uri(self, uri):
        raise NotImplementedError(f"Please implement the `set_tracking_uri` method.")
    
    def get_tracking_uri(self):
        raise NotImplementedError(f"Please implement the `get_tracking_uri` method.")
    
    def get_recorder(self):
        raise NotImplementedError(f"Please implement the `get_recorder` method.")

    def save_object(self, name, data):
        raise NotImplementedError(f"Please implement the `save_object` method.")

    def save_objects(self, name_data_list):
        raise NotImplementedError(f"Please implement the `save_objects` method.")

    def load_object(self, name):
        raise NotImplementedError(f"Please implement the `load_object` method.")
    
    def log_param(self, key, value):
        raise NotImplementedError(f"Please implement the `log_param` method.")

    def log_params(self, params):
        raise NotImplementedError(f"Please implement the `log_params` method.")

    def log_metric(self, key, value, step=None):
        raise NotImplementedError(f"Please implement the `log_metric` method.")

    def log_metrics(self, metrics, step=None):
        raise NotImplementedError(f"Please implement the `log_metrics` method.")
    
    def set_tag(self, key, value):
        raise NotImplementedError(f"Please implement the `set_tag` method.")

    def set_tags(self, tags):
        raise NotImplementedError(f"Please implement the `log_tags` method.")

    def delete_tag(self, key):
        raise NotImplementedError(f"Please implement the `delete_tag` method.")
    
    def log_artifact(self, local_path, artifact_path=None):
        raise NotImplementedError(f"Please implement the `log_artifact` method.")

    def log_artifacts(self, local_dir, artifact_path=None):
        raise NotImplementedError(f"Please implement the `log_artifacts` method.")

    def get_artifact_uri(self, artifact_path=None):
        raise NotImplementedError(f"Please implement the `get_artifact_uri` method.")

class MLflowRecord(Record):
    def __init__(self):
        self.exp_manager = MLflowExpManager()

    @contextmanager
    def start_exp(self, experiment_name=None, uri=None, project_path=None, artifact_location=None, nested=False):
        yield self.exp_manager.start_exp(experiment_name, uri, project_path, artifact_location, nested)
    
    def search_runs(self, experiment_ids=None, filter_string='', run_view_type=1, max_results=100000, order_by=None):
        return self.exp_manager.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by)
    
    def get_exp(self, experiment_id):
        return self.exp_manager.get_exp(experiment_id)

    def get_exp_by_name(self, experiment_name):
        return self.exp_manager.get_exp_by_name(experiment_name)
    
    def create_exp(self, experiment_name, artifact_location=None):
        self.exp_manager.create_exp(experiment_name, artifact_location)
    
    def set_exp(self, experiment_name):
        self.exp_manager.set_exp(experiment_name)

    def delete_exp(self, experiment_id):
        self.exp_manager.delete_exp(experiment_id)

    def set_tracking_uri(self, uri):
        self.exp_manager.set_tracking_uri(uri)
    
    def get_tracking_uri(self):
        return self.exp_manager.get_tracking_uri()
    
    def get_recorder(self):
        return self.exp_manager.get_recorder()

    def save_object(self, name, data):
        self.exp_manager.active_recorder.save_object(name, data)

    def save_objects(self, name_data_list):
        self.exp_manager.active_recorder.save_objects(name_data_list)

    def load_object(self, name):
        return self.exp_manager.active_recorder.load_object(name)
    
    def log_param(self, key, value):
        self.exp_manager.active_recorder.log_param(key, value)

    def log_params(self, params):
        self.exp_manager.active_recorder.log_params(params)

    def log_metric(self, key, value, step=None):
        self.exp_manager.active_recorder.log_metric(key, value, step)

    def log_metrics(self, metrics, step=None):
        self.exp_manager.active_recorder.log_metrics(metrics, step)
    
    def set_tag(self, key, value):
        self.exp_manager.active_recorder.set_tag(key, value)

    def set_tags(self, tags):
        self.exp_manager.active_recorder.set_tags(tags)

    def delete_tag(self, key):
        self.exp_manager.active_recorder.delete_tag(key)
    
    def log_artifact(self, local_path, artifact_path=None):
        self.exp_manager.active_recorder.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        self.exp_manager.active_recorder.log_artifacts(local_dir, artifact_path)

    def get_artifact_uri(self, artifact_path=None):
        return self.exp_manager.active_recorder.get_artifact_uri(artifact_path)

# global record
R = MLflowRecord()