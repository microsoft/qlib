# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse
import importlib

from ... import init
from .config import EstimatorConfigManager
from ...log import get_module_logger
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

args_parser = argparse.ArgumentParser(prog="estimator")
args_parser.add_argument(
    "-c",
    "--config_path",
    required=True,
    type=str,
    help="json config path indicates where to load config.",
)

args = args_parser.parse_args()


class SacredExperiment(object):
    def __init__(
        self,
        experiment_name,
        experiment_dir,
        observer_type="file_storage",
        mongo_url=None,
        db_name=None,
    ):
        """__init__

        :param experiment_name: The name of the experiments.
        :param experiment_dir:  The directory to store all the results of the experiments(This is for file_storage).
        :param observer_type:   The observer to record the results: the `file_storage` or `mongo`
        :param mongo_url:       The mongo url(for mongo observer)
        :param db_name:         The mongo url(for mongo observer)
        """
        self.experiment_name = experiment_name
        self.experiment = Experiment(self.experiment_name)
        self.experiment_dir = experiment_dir
        self.experiment.logger = get_module_logger("Sacred")

        self.observer_type = observer_type
        self.mongo_db_url = mongo_url
        self.mongo_db_name = db_name

        self._setup_experiment()

    def _setup_experiment(self):
        if self.observer_type == "file_storage":
            file_storage_observer = FileStorageObserver.create(basedir=self.experiment_dir)
            self.experiment.observers.append(file_storage_observer)
        elif self.observer_type == "mongo":
            mongo_observer = MongoObserver.create(url=self.mongo_db_url, db_name=self.mongo_db_name)
            self.experiment.observers.append(mongo_observer)
        else:
            raise NotImplementedError("Unsupported observer type: {}".format(self.observer_type))

    def add_artifact(self, filename):
        self.experiment.add_artifact(filename)

    def add_info(self, key, value):
        self.experiment.info[key] = value

    def main_wrapper(self, func):
        return self.experiment.main(func)

    def config_wrapper(self, func):
        return self.experiment.config(func)


CONFIG_MANAGER = EstimatorConfigManager(args.config_path)

ex = SacredExperiment(
    CONFIG_MANAGER.ex_config.name,
    CONFIG_MANAGER.ex_config.sacred_dir,
    observer_type=CONFIG_MANAGER.ex_config.observer_type,
    mongo_url=CONFIG_MANAGER.ex_config.mongo_url,
    db_name=CONFIG_MANAGER.ex_config.db_name,
)

# qlib init
init(
    provider_uri=CONFIG_MANAGER.qlib_data_config.provider_uri,
    mount_path=CONFIG_MANAGER.qlib_data_config.mount_path,
    auto_mount=CONFIG_MANAGER.qlib_data_config.auto_mount,
    region=CONFIG_MANAGER.qlib_data_config.region,
    **CONFIG_MANAGER.qlib_data_config.args
)


@ex.main_wrapper
def _main():
    # 1. Get estimator class.
    estimator_class = getattr(
        importlib.import_module(".estimator", package="qlib.contrib.estimator"),
        "Estimator",
    )
    # 2. Init estimator.
    estimator = estimator_class(CONFIG_MANAGER, ex)
    estimator.run()


def run():
    ex.experiment.run()


if __name__ == "__main__":
    run()
