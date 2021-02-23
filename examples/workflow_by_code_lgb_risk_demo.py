#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
from qlib.config import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data.dataset.handler import DataHandlerLP

import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats.stats import pearsonr
import numpy as np

if __name__ == "__main__":

    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
        from get_data import GetData

        GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    ###################################
    # train model
    ###################################
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
        "infer_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "ZScoreNorm", "kwargs": {"fields_group": "feature"}},
            {"class": "Fillna", "kwargs": {}},
        ],
        "learn_processors": [{
            "class": "DropnaLabel", },
        ],
        "label": (["Ref(Min($low, 5), -4)/$close - 1"], ["LABEL0"])  # the period for risk prediction is 5 days
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8999,
                "learning_rate": 0.02,
                "subsample": 0.7,
                "lambda_l1": 11.9668,
                "lambda_l2": 339.1301,
                "max_depth": 16,
                "num_leaves": 31,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha360",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    }

    port_analysis_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.strategy",
            "kwargs": {
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "verbose": False,
            "limit_threshold": 0.095,
            "account": 100000000,
            "benchmark": benchmark,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
            "return_order": True,
        },
    }

    # model initiaiton
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())

    def heatmap(actual_risk, predicted_risk, step=0.02):
        """
        plot the precision heatmap as a visualized evaluation for risk predition
        :param actual_risk: the LABEL0 of test samples
        :param predicted_risk: the predicted results of test samples
        :param step: the internal size of risk values on axis
        :return:
        """
        num_step = math.ceil(-predicted_risk.min() / step)
        matrix = np.zeros((num_step, num_step), dtype=np.float)
        for pred_thresh in range(num_step):
            for act_thresh in range(num_step):
                actual_positive = actual_risk < -act_thresh*step
                predicted_alarm = predicted_risk < -pred_thresh*step
                num_alarm = predicted_alarm.sum()
                num_tp = (actual_positive & predicted_alarm).sum()
                matrix[pred_thresh, act_thresh] = num_tp / num_alarm
        axis_labels = ['{:.3f}'.format(-x * step) for x in range(num_step)]
        return matrix, axis_labels

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)

        # prediction
        actual_risk = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_I)['LABEL0']
        pred = model.predict(dataset)

        result_df = pd.concat((actual_risk, pred), axis=1)
        result_df.columns = ['Actual Risk', 'Predicted Risk']
        result_df.dropna(inplace=True)
        actual_risk, predicted_risk = result_df.iloc[:, 0], result_df.iloc[:, 1]
        corr = pearsonr(actual_risk, predicted_risk)[0]
        print('The correlation between predicted risk and actual risk is: {:.6f}'.format(corr))

        # visualized results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        sns.histplot(actual_risk, ax=axes[0, 0])
        axes[0, 0].set_title('Market: {}  Actual Risk'.format(market))
        axes[0, 0].grid()

        sns.histplot(predicted_risk, ax=axes[0, 1])
        axes[0, 1].set_title('Feature: {}  Predicted Risk'.format(task['dataset']['kwargs']['handler']['class']))
        axes[0, 1].grid()

        sns.scatterplot(data=result_df, ax=axes[1, 0], x='Actual Risk', y='Predicted Risk', s=20)
        axes[1, 0].set_title('Market: {}  Feature: {}  Corr: {:.5f}'.format(
            market, task['dataset']['kwargs']['handler']['class'], corr))
        axes[1, 0].grid()

        matrix, ax_labels = heatmap(actual_risk, predicted_risk)
        sns.heatmap(matrix, annot=True, fmt=".3f", xticklabels=ax_labels, yticklabels=ax_labels, ax=axes[1, 1],
                    )
        axes[1, 1].set_xlabel('Predicted Alarm Threshold')
        axes[1, 1].set_ylabel('Actual Positive Threshold')
        axes[1, 1].set_title('Risk Prediction Precision Heatmap')
        plt.show()
