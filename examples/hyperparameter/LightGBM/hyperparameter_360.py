import qlib
import optuna
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_MARKET, CSI300_BENCH, BR_MARKET, BR_BENCH, DATASET_ALPHA360_CLASS
from qlib.log import get_module_logger

logger = get_module_logger("Hyperparameter")

## TODO: improve this function beacuse it isn't  working correctly. Configuration is being setup manually
# DATASET_CONFIG = get_dataset_config(market=CSI300_MARKET, dataset_class=DATASET_ALPHA360_CLASS)

market = CSI300_MARKET
benchmark = CSI300_BENCH

## For brazilian market
# market = BR_MARKET
# benchmark = BR_BENCH

data_handler_config = {
    "start_time": "2007-01-01",
    "end_time": "2019-12-31",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2016-12-31",
    "instruments": market,
    "infer_processors": [],
    "learn_processors": [
      {
        "class": "DropnaLabel"
      },
      {
        "class": "CSRankNorm",
        "kwargs": {
          "fields_group": "label"
        }
      }
    ],
    "label": [
      "(Ref($close, -1) / $close) - 1"
    ]
}

dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config,
        },
        "segments": {
            "train": ("2007-01-01", "2016-12-31"),
            "valid": ("2017-01-01", "2017-12-31"),
            "test": ("2018-01-01", "2019-12-31"),
        },
    },
}


def objective(trial):
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 1),
                "subsample": trial.suggest_uniform("subsample", 0, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
                "max_depth": 10,
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }

    logger.info("model:\n{:}".format(task["model"]))
    evals_result = dict()
    model = init_instance_by_config(task["model"])
    model.fit(dataset, evals_result=evals_result)
    print("---- evals_result[valid] ------")
    print(evals_result["valid"]["l2"])
    return min(evals_result["valid"]["l2"])


if __name__ == "__main__":
    logger.info("Qlib intialization")
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri)

    logger.info("Dataset intialization")
    dataset = init_instance_by_config(dataset_config)

    logger.info("Start parameter tuning")
    study = optuna.Study(study_name="LGBM_360", storage="sqlite:///db.sqlite3")
    study.optimize(objective)
