# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import yaml
import copy
import os


class TunerConfigManager:
    def __init__(self, config_path):

        if not config_path:
            raise ValueError("Config path is invalid.")
        self.config_path = config_path

        with open(config_path) as fp:
            config = yaml.safe_load(fp)
        self.config = copy.deepcopy(config)

        self.pipeline_ex_config = PipelineExperimentConfig(config.get("experiment", dict()), self)
        self.pipeline_config = config.get("tuner_pipeline", list())
        self.optim_config = OptimizationConfig(config.get("optimization_criteria", dict()), self)

        self.time_config = config.get("time_period", dict())
        self.data_config = config.get("data", dict())
        self.backtest_config = config.get("backtest", dict())
        self.qlib_client_config = config.get("qlib_client", dict())


class PipelineExperimentConfig:
    def __init__(self, config, TUNER_CONFIG_MANAGER):
        """
        :param config:  The config dict for tuner experiment
        :param TUNER_CONFIG_MANAGER:   The tuner config manager
        """
        self.name = config.get("name", "tuner_experiment")
        # The dir of the config
        self.global_dir = config.get("dir", os.path.dirname(TUNER_CONFIG_MANAGER.config_path))
        # The dir of the result of tuner experiment
        self.tuner_ex_dir = config.get("tuner_ex_dir", os.path.join(self.global_dir, self.name))
        if not os.path.exists(self.tuner_ex_dir):
            os.makedirs(self.tuner_ex_dir)
        # The dir of the results of all estimator experiments
        self.estimator_ex_dir = config.get("estimator_ex_dir", os.path.join(self.tuner_ex_dir, "estimator_experiment"))
        if not os.path.exists(self.estimator_ex_dir):
            os.makedirs(self.estimator_ex_dir)
        # Get the tuner type
        self.tuner_module_path = config.get("tuner_module_path", "qlib.contrib.tuner.tuner")
        self.tuner_class = config.get("tuner_class", "QLibTuner")
        # Save the tuner experiment for further view
        tuner_ex_config_path = os.path.join(self.tuner_ex_dir, "tuner_config.yaml")
        with open(tuner_ex_config_path, "w") as fp:
            yaml.dump(TUNER_CONFIG_MANAGER.config, fp)


class OptimizationConfig:
    def __init__(self, config, TUNER_CONFIG_MANAGER):

        self.report_type = config.get("report_type", "pred_long")
        if self.report_type not in [
            "pred_long",
            "pred_long_short",
            "pred_short",
            "excess_return_without_cost",
            "excess_return_with_cost",
            "model",
        ]:
            raise ValueError(
                "report_type should be one of pred_long, pred_long_short, pred_short, excess_return_without_cost, excess_return_with_cost and model"
            )

        self.report_factor = config.get("report_factor", "information_ratio")
        if self.report_factor not in [
            "annualized_return",
            "information_ratio",
            "max_drawdown",
            "mean",
            "std",
            "model_score",
            "model_pearsonr",
        ]:
            raise ValueError(
                "report_factor should be one of annualized_return, information_ratio, max_drawdown, mean, std, model_pearsonr and model_score"
            )

        self.optim_type = config.get("optim_type", "max")
        if self.optim_type not in ["min", "max", "correlation"]:
            raise ValueError("optim_type should be min, max or correlation")
