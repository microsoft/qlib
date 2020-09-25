# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8

import pandas as pd

import os
import copy
import json
import yaml
import pickle

import qlib
from ..evaluate import risk_analysis
from ..evaluate import backtest as normal_backtest
from ..evaluate import long_short_backtest
from .config import ExperimentConfig
from .fetcher import create_fetcher_with_config

from ...log import get_module_logger, TimeInspector
from ...utils import get_module_by_module_path, compare_dict_value


class Estimator(object):
    def __init__(self, config_manager, sacred_ex):

        # Set logger.
        self.logger = get_module_logger("Estimator")

        # 1. Set config manager.
        self.config_manager = config_manager

        # 2. Set configs.
        self.ex_config = config_manager.ex_config
        self.data_config = config_manager.data_config
        self.model_config = config_manager.model_config
        self.trainer_config = config_manager.trainer_config
        self.strategy_config = config_manager.strategy_config
        self.backtest_config = config_manager.backtest_config

        # If experiment.mode is test or experiment.finetune is True, load the experimental results in the loader
        if self.ex_config.mode == self.ex_config.TEST_MODE or self.ex_config.finetune:
            self.compare_config_with_config_manger(self.config_manager)

        # 3. Set sacred_experiment.
        self.ex = sacred_ex

        # 4. Init data handler.
        self.data_handler = None
        self._init_data_handler()

        # 5. Init trainer.
        self.trainer = None
        self._init_trainer()

        # 6. Init strategy.
        self.strategy = None
        self._init_strategy()

    def _init_data_handler(self):
        handler_module = get_module_by_module_path(self.data_config.handler_module_path)

        # Set market
        market = self.data_config.handler_filter.get("market", None)
        if market is None:
            if "market" in self.data_config.handler_parameters:
                self.logger.warning(
                    "Warning: The market in data.args section is deprecated. "
                    "It only works when market is not set in data.filter section. "
                    "It will be overridden by market in the data.filter section."
                )
                market = self.data_config.handler_parameters["market"]
            else:
                market = "csi500"

        self.data_config.handler_parameters["market"] = market

        data_filter_list = []
        handler_filters = self.data_config.handler_filter.get("filter_pipeline", list())
        for h_filter in handler_filters:
            filter_module_path = h_filter.get("module_path", "qlib.data.filter")
            filter_class_name = h_filter.get("class", "")
            filter_parameters = h_filter.get("args", {})
            filter_module = get_module_by_module_path(filter_module_path)
            filter_class = getattr(filter_module, filter_class_name)
            data_filter = filter_class(**filter_parameters)
            data_filter_list.append(data_filter)

        self.data_config.handler_parameters["data_filter_list"] = data_filter_list
        handler_class = getattr(handler_module, self.data_config.handler_class)
        self.data_handler = handler_class(**self.data_config.handler_parameters)

    def _init_trainer(self):

        model_module = get_module_by_module_path(self.model_config.model_module_path)
        trainer_module = get_module_by_module_path(self.trainer_config.trainer_module_path)
        model_class = getattr(model_module, self.model_config.model_class)
        trainer_class = getattr(trainer_module, self.trainer_config.trainer_class)

        self.trainer = trainer_class(
            model_class,
            self.model_config.save_path,
            self.model_config.parameters,
            self.data_handler,
            self.ex,
            **self.trainer_config.parameters
        )

    def _init_strategy(self):

        module = get_module_by_module_path(self.strategy_config.strategy_module_path)
        strategy_class = getattr(module, self.strategy_config.strategy_class)
        self.strategy = strategy_class(**self.strategy_config.parameters)

    def run(self):
        if self.ex_config.mode == ExperimentConfig.TRAIN_MODE:
            self.trainer.train()
        elif self.ex_config.mode == ExperimentConfig.TEST_MODE:
            self.trainer.load()
        else:
            raise ValueError("unexpected mode: %s" % self.ex_config.mode)
        analysis = self.backtest()
        print(analysis)
        self.logger.info(
            "experiment id: {}, experiment name: {}".format(self.ex.experiment.current_run._id, self.ex_config.name)
        )

        # Remove temp dir
        # shutil.rmtree(self.ex_config.tmp_run_dir)

    def backtest(self):
        TimeInspector.set_time_mark()
        # 1. Get pred and prediction score of model(s).
        pred = self.trainer.get_test_score()
        try:
            performance = self.trainer.get_test_performance()
        except NotImplementedError:
            performance = None
        # 2. Normal Backtest.
        report_normal, positions_normal = self._normal_backtest(pred)
        # 3. Long-Short Backtest.
        # Deprecated
        # long_short_reports = self._long_short_backtest(pred)
        # 4. Analyze
        analysis_df = self._analyze(report_normal)
        # 5. Save.
        self._save_backtest_result(
            pred,
            analysis_df,
            positions_normal,
            report_normal,
            # long_short_reports,
            performance,
        )
        return analysis_df

    def _normal_backtest(self, pred):
        TimeInspector.set_time_mark()
        if "account" not in self.backtest_config.normal_backtest_parameters:
            if "account" in self.strategy_config.parameters:
                self.logger.warning(
                    "Warning: The account in strategy section is deprecated. "
                    "It only works when account is not set in backtest section. "
                    "It will be overridden by account in the backtest section."
                )
                self.backtest_config.normal_backtest_parameters["account"] = self.strategy_config.parameters["account"]
        report_normal, positions_normal = normal_backtest(
            pred, strategy=self.strategy, **self.backtest_config.normal_backtest_parameters
        )
        TimeInspector.log_cost_time("Finished normal backtest.")
        return report_normal, positions_normal

    def _long_short_backtest(self, pred):
        TimeInspector.set_time_mark()
        long_short_reports = long_short_backtest(pred, **self.backtest_config.long_short_backtest_parameters)
        TimeInspector.log_cost_time("Finished long-short backtest.")
        return long_short_reports

    @staticmethod
    def _analyze(report_normal):
        TimeInspector.set_time_mark()

        analysis = dict()
        # analysis["pred_long"] = risk_analysis(long_short_reports["long"])
        # analysis["pred_short"] = risk_analysis(long_short_reports["short"])
        # analysis["pred_long_short"] = risk_analysis(long_short_reports["long_short"])
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])
        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        TimeInspector.log_cost_time(
            "Finished generating analysis," " average turnover is: {0:.4f}.".format(report_normal["turnover"].mean())
        )
        return analysis_df

    def _save_backtest_result(self, pred, analysis, positions, report_normal, performance):
        # 1. Result dir.
        result_dir = os.path.join(self.config_manager.ex_config.tmp_run_dir, "result")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.ex.add_info(
            "task_config",
            json.loads(json.dumps(self.config_manager.config, default=str)),
        )

        # 2. Pred.
        TimeInspector.set_time_mark()
        pred_pkl_path = os.path.join(result_dir, "pred.pkl")
        pred.to_pickle(pred_pkl_path)
        self.ex.add_artifact(pred_pkl_path)
        TimeInspector.log_cost_time("Finished saving pred.pkl to: {}".format(pred_pkl_path))

        # 3. Ana.
        TimeInspector.set_time_mark()
        analysis_pkl_path = os.path.join(result_dir, "analysis.pkl")
        analysis.to_pickle(analysis_pkl_path)
        self.ex.add_artifact(analysis_pkl_path)
        TimeInspector.log_cost_time("Finished saving analysis.pkl to: {}".format(analysis_pkl_path))

        # 4. Pos.
        TimeInspector.set_time_mark()
        positions_pkl_path = os.path.join(result_dir, "positions.pkl")
        with open(positions_pkl_path, "wb") as fp:
            pickle.dump(positions, fp)
        self.ex.add_artifact(positions_pkl_path)
        TimeInspector.log_cost_time("Finished saving positions.pkl to: {}".format(positions_pkl_path))

        # 5. Report normal.
        TimeInspector.set_time_mark()
        report_normal_pkl_path = os.path.join(result_dir, "report_normal.pkl")
        report_normal.to_pickle(report_normal_pkl_path)
        self.ex.add_artifact(report_normal_pkl_path)
        TimeInspector.log_cost_time("Finished saving report_normal.pkl to: {}".format(report_normal_pkl_path))

        # 6. Report long short.
        # Deprecated
        # for k, name in zip(
        #     ["long", "short", "long_short"],
        #     ["report_long.pkl", "report_short.pkl", "report_long_short.pkl"],
        # ):
        #     TimeInspector.set_time_mark()
        #     pkl_path = os.path.join(result_dir, name)
        #     long_short_reports[k].to_pickle(pkl_path)
        #     self.ex.add_artifact(pkl_path)
        #     TimeInspector.log_cost_time("Finished saving {} to: {}".format(name, pkl_path))

        # 7. Origin test label.
        TimeInspector.set_time_mark()
        label_pkl_path = os.path.join(result_dir, "label.pkl")
        self.data_handler.get_origin_test_label_with_date(
            self.trainer_config.parameters["test_start_date"],
            self.trainer_config.parameters["test_end_date"],
        ).to_pickle(label_pkl_path)
        self.ex.add_artifact(label_pkl_path)
        TimeInspector.log_cost_time("Finished saving label.pkl to: {}".format(label_pkl_path))

        # 8. Experiment info, save the model(s) performance here.
        TimeInspector.set_time_mark()
        cur_ex_id = self.ex.experiment.current_run._id
        exp_info = {
            "id": cur_ex_id,
            "name": self.ex_config.name,
            "performance": performance,
            "observer_type": self.ex_config.observer_type,
        }

        if self.ex_config.observer_type == ExperimentConfig.OBSERVER_MONGO:
            exp_info.update(
                {
                    "mongo_url": self.ex_config.mongo_url,
                    "db_name": self.ex_config.db_name,
                }
            )
        else:
            exp_info.update({"dir": self.ex_config.global_dir})

        with open(self.ex_config.exp_info_path, "w") as fp:
            json.dump(exp_info, fp, indent=4, sort_keys=True)
        self.ex.add_artifact(self.ex_config.exp_info_path)
        TimeInspector.log_cost_time("Finished saving ex_info to: {}".format(self.ex_config.exp_info_path))

    @staticmethod
    def compare_config_with_config_manger(config_manager):
        """Compare loader model args and current config with ConfigManage

        :param config_manager: ConfigManager
        :return:
        """
        fetcher = create_fetcher_with_config(config_manager, load_form_loader=True)
        loader_mode_config = fetcher.get_experiment(
            exp_name=config_manager.ex_config.loader_name,
            exp_id=config_manager.ex_config.loader_id,
            fields=["task_config"],
        )["task_config"]
        with open(config_manager.config_path) as fp:
            current_config = yaml.load(fp.read())
            current_config = json.loads(json.dumps(current_config, default=str))

        logger = get_module_logger("Estimator")

        loader_mode_config = copy.deepcopy(loader_mode_config)
        current_config = copy.deepcopy(current_config)

        # Require test_mode_config.test_start_date <= current_config.test_start_date
        loader_trainer_args = loader_mode_config.get("trainer", {}).get("args", {})
        cur_trainer_args = current_config.get("trainer", {}).get("args", {})
        loader_start_date = loader_trainer_args.pop("test_start_date")
        cur_test_start_date = cur_trainer_args.pop("test_start_date")
        assert (
            loader_start_date <= cur_test_start_date
        ), "Require: loader_mode_config.test_start_date <= current_config.test_start_date"

        # TODO: For the user's own extended `Trainer`, the support is not very good
        if "RollingTrainer" == current_config.get("trainer", {}).get("class", None):
            loader_period = loader_trainer_args.pop("rolling_period")
            cur_period = cur_trainer_args.pop("rolling_period")
            assert (
                loader_period == cur_period
            ), "Require: loader_mode_config.rolling_period == current_config.rolling_period"

        compare_section = ["trainer", "model", "data"]
        for section in compare_section:
            changes = compare_dict_value(loader_mode_config.get(section, {}), current_config.get(section, {}))
            if changes:
                logger.warning("Warning: Loader mode config and current config, `{}` are different:\n".format(section))
