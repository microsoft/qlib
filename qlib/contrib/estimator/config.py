# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import yaml
import copy
import os
import json
import tempfile
from pathlib import Path
from ...config import REG_CN


class EstimatorConfigManager(object):
    def __init__(self, config_path):

        if not config_path:
            raise ValueError("Config path is invalid.")
        self.config_path = config_path

        with open(config_path) as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
        self.config = copy.deepcopy(config)

        self.ex_config = ExperimentConfig(config.get("experiment", dict()), self)
        self.data_config = DataConfig(config.get("data", dict()), self)
        self.model_config = ModelConfig(config.get("model", dict()), self)
        self.trainer_config = TrainerConfig(config.get("trainer", dict()), self)
        self.strategy_config = StrategyConfig(config.get("strategy", dict()), self)
        self.backtest_config = BacktestConfig(config.get("backtest", dict()), self)
        self.qlib_data_config = QlibDataConfig(config.get("qlib_data", dict()), self)

        # If the start_date and end_date are not given in data_config, they will be referred from the trainer_config.
        handler_start_date = self.data_config.handler_parameters.get("start_date", None)
        handler_end_date = self.data_config.handler_parameters.get("end_date", None)
        if handler_start_date is None:
            self.data_config.handler_parameters["start_date"] = self.trainer_config.parameters["train_start_date"]
        if handler_end_date is None:
            self.data_config.handler_parameters["end_date"] = self.trainer_config.parameters["test_end_date"]


class ExperimentConfig(object):
    TRAIN_MODE = "train"
    TEST_MODE = "test"

    OBSERVER_FILE_STORAGE = "file_storage"
    OBSERVER_MONGO = "mongo"

    def __init__(self, config, CONFIG_MANAGER):
        """__init__

        :param config:         The config dict for experiment
        :param CONFIG_MANAGER: The estimator config manager
        """
        self.name = config.get("name", "test_experiment")
        # The dir of the result of  all the experiments
        self.global_dir = config.get("dir", os.path.dirname(CONFIG_MANAGER.config_path))
        # The dir of the result of current experiment
        self.ex_dir = os.path.join(self.global_dir, self.name)
        if not os.path.exists(self.ex_dir):
            os.makedirs(self.ex_dir)
        self.tmp_run_dir = tempfile.mkdtemp(dir=self.ex_dir)
        self.mode = config.get("mode", ExperimentConfig.TRAIN_MODE)
        self.sacred_dir = os.path.join(self.ex_dir, "sacred")
        self.observer_type = config.get("observer_type", ExperimentConfig.OBSERVER_FILE_STORAGE)
        self.mongo_url = config.get("mongo_url", None)
        self.db_name = config.get("db_name", None)
        self.finetune = config.get("finetune", False)

        # The path of the experiment id of the experiment
        self.exp_info_path = config.get("exp_info_path", os.path.join(self.ex_dir, "exp_info.json"))
        exp_info_dir = Path(self.exp_info_path).parent
        exp_info_dir.mkdir(parents=True, exist_ok=True)

        # Test mode config
        loader_args = config.get("loader", dict())
        if self.mode == ExperimentConfig.TEST_MODE or self.finetune:
            loader_exp_info_path = loader_args.get("exp_info_path", None)
            self.loader_model_index = loader_args.get("model_index", None)
            if (loader_exp_info_path is not None) and (os.path.exists(loader_exp_info_path)):
                with open(loader_exp_info_path) as fp:
                    loader_dict = json.load(fp)
                    for k, v in loader_dict.items():
                        setattr(self, "loader_{}".format(k), v)
                        # Check loader experiment id
                assert hasattr(self, "loader_id"), "If mode is test or finetune is True, loader must contain id."
            else:
                self.loader_id = loader_args.get("id", None)
                if self.loader_id is None:
                    raise ValueError("If mode is test or finetune is True, loader must contain id.")

                self.loader_observer_type = loader_args.get("observer_type", self.observer_type)
                self.loader_name = loader_args.get("name", self.name)
                self.loader_dir = loader_args.get("dir", self.global_dir)

                self.loader_mongo_url = loader_args.get("mongo_url", self.mongo_url)
                self.loader_db_name = loader_args.get("db_name", self.db_name)


class DataConfig(object):
    def __init__(self, config, CONFIG_MANAGER):
        """__init__

        :param config:         The config dict for data
        :param CONFIG_MANAGER: The estimator config manager
        """
        self.handler_module_path = config.get("module_path", "qlib.contrib.estimator.handler")
        self.handler_class = config.get("class", "ALPHA360")
        self.handler_parameters = config.get("args", dict())
        self.handler_filter = config.get("filter", dict())
        # Update provider uri.


class ModelConfig(object):
    def __init__(self, config, CONFIG_MANAGER):
        """__init__

        :param config:         The config dict for model
        :param CONFIG_MANAGER: The estimator config manager
        """
        self.model_class = config.get("class", "Model")
        self.model_module_path = config.get("module_path", "qlib.contrib.model")
        self.save_dir = os.path.join(CONFIG_MANAGER.ex_config.tmp_run_dir, "model")
        self.save_path = config.get("save_path", os.path.join(self.save_dir, "model.bin"))
        self.parameters = config.get("args", dict())
        # Make dir if need.
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


class TrainerConfig(object):
    def __init__(self, config, CONFIG_MANAGER):
        """__init__

        :param config:         The config dict for trainer
        :param CONFIG_MANAGER: The estimator config manager
        """
        self.trainer_class = config.get("class", "StaticTrainer")
        self.trainer_module_path = config.get("module_path", "qlib.contrib.estimator.trainer")
        self.parameters = config.get("args", dict())


class StrategyConfig(object):
    def __init__(self, config, CONFIG_MANAGER):
        """__init__

        :param config:         The config dict for strategy
        :param CONFIG_MANAGER: The estimator config manager
        """
        self.strategy_class = config.get("class", "TopkDropoutStrategy")
        self.strategy_module_path = config.get("module_path", "qlib.contrib.strategy.strategy")
        self.parameters = config.get("args", dict())


class BacktestConfig(object):
    def __init__(self, config, CONFIG_MANAGE):
        """__init__

        :param config:          The config dict for strategy
        :param CONFIG_MANAGE:   The estimator config manager
        """
        self.normal_backtest_parameters = config.get("normal_backtest_args", dict())
        self.long_short_backtest_parameters = config.get("long_short_backtest_args", dict())


class QlibDataConfig(object):
    def __init__(self, config, CONFIG_MANAGE):
        """__init__

        :param config:          The config dict for qlib_client
        :param CONFIG_MANAGE:   The estimator config manager
        """
        self.provider_uri = config.pop("provider_uri", "~/.qlib/qlib_data/cn_data")
        self.auto_mount = config.pop("auto_mount", False)
        self.mount_path = config.pop("mount_path", "~/.qlib/qlib_data/cn_data")
        self.region = config.pop("region", REG_CN)
        self.args = config
