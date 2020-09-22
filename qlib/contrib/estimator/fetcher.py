# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8

import copy
import json
import yaml
import pickle
import gridfs
import pymongo
from pathlib import Path
from abc import abstractmethod

from .config import EstimatorConfigManager, ExperimentConfig


class Fetcher(object):
    """Sacred Experiments Fetcher"""

    @abstractmethod
    def _get_experiment(self, exp_name, exp_id):
        """Get experiment basic info with experiment and experiment id

        :param exp_name: experiment name
        :param exp_id: experiment id
        :return: dict
            Must contain keys: _id, experiment, info, stop_time.
            Here is an example below for FileFetcher.
            exp = {
                '_id': exp_id,                              # experiment id
                'path': path,                               # experiment result path
                'experiment': {'name': exp_name},           # experiment
                'info': info,                               # experiment config info
                'stop_time': run.get('stop_time', None)     # The time the experiment ended
            }

        """
        pass

    @abstractmethod
    def _list_experiments(self, exp_name=None):
        """Get experiment basic info list with experiment name

        :param exp_name: experiment name
        :return: list

        """
        pass

    @abstractmethod
    def _iter_artifacts(self, experiment):
        """Get information about the data in the experiment results

        :param experiment: `self._get_experiment` method result
        :return: iterable
            Each element contains two elements.
                first element  : data name
                second element : data uri
        """
        pass

    @abstractmethod
    def _load_data(self, uri):
        """Load data with uri

        :param uri: data uri
        :return: bytes
        """
        pass

    @staticmethod
    def model_dict_to_buffer_list(model_dict):
        """

        :param model_dict:
        :return:
        """
        model_list = []
        is_static_model = False
        if len(model_dict) == 1 and list(model_dict.keys())[0] == "model.bin":
            is_static_model = True
            model_list.append(list(model_dict.values())[0])
        else:
            sep = "model.bin_"
            model_ids = list(map(lambda x: int(x.split(sep)[1]), model_dict.keys()))
            min_id, max_id = min(model_ids), max(model_ids)
            for i in range(min_id, max_id + 1):
                model_key = sep + str(i)
                model = model_dict.get(model_key, None)
                if model is None:
                    print(
                        "WARNING: In Fetcher, {} is missing when the get model is in the get_experiment function.".format(
                            model_key
                        )
                    )
                    break
                else:
                    model_list.append(model)

        if is_static_model:
            return model_list[0]

        return model_list

    def get_experiments(self, exp_name=None):
        """Get experiments with name.

        :param exp_name: str
            If `exp_name` is set to None, then all experiments will return.
        :return: dict
            Experiments info dict(Including experiment id and task_config to run the
            experiment). Here is an example below.
            {
                'a_experiment': [
                    {
                        'id': '1',
                        'task_config': {...}
                    },
                    ...
                ]
                ...
            }
        """
        res = dict()
        for ex in self._list_experiments(exp_name):
            name = ex["experiment"]["name"]
            tmp = {
                "id": ex["_id"],
                "task_config": ex["info"].get("task_config", {}),
                "ex_run_stop_time": ex.get("stop_time", None),
            }
            res.setdefault(name, []).append(tmp)
        return res

    def get_experiment(self, exp_name, exp_id, fields=None):
        """

        :param exp_name:
        :param exp_id:
        :param fields: list
            Experiment result fields, if fields is None, will get all fields.
                Currently supported fields:
                    ['model', 'analysis', 'positions', 'report_normal', 'pred', 'task_config', 'label']
        :return: dict
        """
        fields = copy.copy(fields)
        ex = self._get_experiment(exp_name, exp_id)
        results = dict()
        model_dict = dict()
        for name, uri in self._iter_artifacts(ex):
            # When saving, use `sacred.experiment.add_artifact(filename)` , so `name` is os.path.basename(filename)
            prefix = name.split(".")[0]
            if fields and prefix not in fields:
                continue
            data = self._load_data(uri)
            if prefix == "model":
                model_dict[name] = data
            else:
                results[prefix] = pickle.loads(data)
        # Sort model
        if model_dict:
            results["model"] = self.model_dict_to_buffer_list(model_dict)

        # Info
        results["task_config"] = ex["info"].get("task_config", {})
        return results

    def estimator_config_to_dict(self, exp_name, exp_id):
        """Save configuration to file

        :param exp_name:
        :param exp_id:
        :return: config dict
        """

        return self.get_experiment(exp_name, exp_id, fields=["task_config"])["task_config"]


class FileFetcher(Fetcher):
    """File Fetcher"""

    def __init__(self, experiments_dir):
        self.experiments_dir = Path(experiments_dir)

    def _get_experiment(self, exp_name, exp_id):
        path = self.experiments_dir / exp_name / "sacred" / str(exp_id)
        info_path = path / "info.json"
        run_path = path / "run.json"

        if info_path.exists():
            with info_path.open("r") as f:
                info = json.load(f)
        else:
            info = {}

        if run_path.exists():
            with run_path.open("r") as f:
                run = json.load(f)
        else:
            run = {}

        exp = {
            "_id": exp_id,
            "path": path,
            "experiment": {"name": exp_name},
            "info": info,
            "stop_time": run.get("stop_time", None),
        }
        return exp

    def _list_experiments(self, exp_name=None):
        runs = []
        for path in self.experiments_dir.glob("{}/sacred/[!_]*".format(exp_name or "*")):
            exp_name, exp_id = path.parents[1].name, path.name
            runs.append(self._get_experiment(exp_name, exp_id))
        return runs

    def _iter_artifacts(self, experiment):
        if experiment is None:
            return []

        for fname in experiment["path"].iterdir():
            if fname.suffix == ".pkl" or ".bin" in fname.suffix:
                name, uri = fname.name, str(fname)
                yield name, uri

    def _load_data(self, uri):
        with open(uri, "rb") as f:
            data = f.read()
        return data


class MongoFetcher(Fetcher):
    """MongoDB Fetcher"""

    def __init__(self, mongo_url, db_name):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = None
        self.db = None
        self.runs = None
        self.fs = None
        self._setup_mongo_client()

    def _setup_mongo_client(self):
        self.client = pymongo.MongoClient(self.mongo_url)
        self.db = self.client[self.db_name]
        self.runs = self.db.runs
        self.fs = gridfs.GridFS(self.db)

    def _get_experiment(self, exp_name, exp_id):
        return self.runs.find_one({"_id": exp_id})

    def _list_experiments(self, exp_name=None):
        if exp_name is None:
            return self.runs.find()
        return self.runs.find({"experiment.name": exp_name})

    def _iter_artifacts(self, experiment):
        if experiment is None:
            return []
        for artifact in experiment.get("artifacts", []):
            name, uri = artifact["name"], artifact["file_id"]
            yield name, uri

    def _load_data(self, uri):
        data = self.fs.get(uri).read()
        return data


def create_fetcher_with_config(config_manager: EstimatorConfigManager, load_form_loader: bool = False):
    """Create fetcher with loader config

    :param config_manager:
    :param load_form_loader
    :return:
    """
    flag = ""
    if load_form_loader:
        flag = "loader_"
    if config_manager.ex_config.observer_type == ExperimentConfig.OBSERVER_FILE_STORAGE:
        return FileFetcher(eval("config_manager.ex_config.{}_dir".format("loader" if load_form_loader else "global")))
    elif config_manager.ex_config.observer_type == ExperimentConfig.OBSERVER_MONGO:
        return MongoFetcher(
            mongo_url=eval("config_manager.ex_config.{}mongo_url".format(flag)),
            db_name=eval("config_manager.ex_config.{}db_name".format(flag)),
        )
    else:
        return NotImplementedError("Unkown Backend")
