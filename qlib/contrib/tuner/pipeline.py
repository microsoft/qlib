# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import logging
import importlib
from abc import abstractmethod

from ...log import get_module_logger, TimeInspector
from ...utils import get_module_by_module_path


class Pipeline:

    GLOBAL_BEST_PARAMS_NAME = "global_best_params.json"

    def __init__(self, tuner_config_manager):

        self.logger = get_module_logger("Pipeline", sh_level=logging.INFO)

        self.tuner_config_manager = tuner_config_manager

        self.pipeline_ex_config = tuner_config_manager.pipeline_ex_config
        self.optim_config = tuner_config_manager.optim_config
        self.time_config = tuner_config_manager.time_config
        self.pipeline_config = tuner_config_manager.pipeline_config
        self.data_config = tuner_config_manager.data_config
        self.backtest_config = tuner_config_manager.backtest_config
        self.qlib_client_config = tuner_config_manager.qlib_client_config

        self.global_best_res = None
        self.global_best_params = None
        self.best_tuner_index = None

    def run(self):

        TimeInspector.set_time_mark()
        for tuner_index, tuner_config in enumerate(self.pipeline_config):
            tuner = self.init_tuner(tuner_index, tuner_config)
            tuner.tune()
            if self.global_best_res is None or self.global_best_res > tuner.best_res:
                self.global_best_res = tuner.best_res
                self.global_best_params = tuner.best_params
                self.best_tuner_index = tuner_index
        TimeInspector.log_cost_time("Finished tuner pipeline.")

        self.save_tuner_exp_info()

    def init_tuner(self, tuner_index, tuner_config):
        """
        Implement this method to build the tuner by config
        return: tuner
        """
        # 1. Add experiment config in tuner_config
        tuner_config["experiment"] = {
            "name": "estimator_experiment_{}".format(tuner_index),
            "id": tuner_index,
            "dir": self.pipeline_ex_config.estimator_ex_dir,
            "observer_type": "file_storage",
        }
        tuner_config["qlib_client"] = self.qlib_client_config
        # 2. Add data config in tuner_config
        tuner_config["data"] = self.data_config
        # 3. Add backtest config in tuner_config
        tuner_config["backtest"] = self.backtest_config
        # 4. Update trainer in tuner_config
        tuner_config["trainer"].update({"args": self.time_config})

        # 5. Import Tuner class
        tuner_module = get_module_by_module_path(self.pipeline_ex_config.tuner_module_path)
        tuner_class = getattr(tuner_module, self.pipeline_ex_config.tuner_class)
        # 6. Return the specific tuner
        return tuner_class(tuner_config, self.optim_config)

    def save_tuner_exp_info(self):

        TimeInspector.set_time_mark()
        save_path = os.path.join(self.pipeline_ex_config.tuner_ex_dir, Pipeline.GLOBAL_BEST_PARAMS_NAME)
        with open(save_path, "w") as fp:
            json.dump(self.global_best_params, fp)
        TimeInspector.log_cost_time("Finished save global best tuner parameters.")

        self.logger.info("Best Tuner id: {}.".format(self.best_tuner_index))
        self.logger.info("Global best parameters: {}.".format(self.global_best_params))
        self.logger.info("You can check the best parameters at {}.".format(save_path))
