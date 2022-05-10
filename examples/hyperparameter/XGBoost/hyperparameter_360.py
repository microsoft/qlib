
import qlib
import optuna
from qlib.constant import REG_US 
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_MARKET, CSI300_BENCH, BR_MARKET, BR_BENCH, DATASET_ALPHA360_CLASS
from qlib.log import get_module_logger

logger = get_module_logger("Hyperparameter")

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
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "eval_metric": "rmse",
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
                "n_estimators": trial.suggest_int("n_estimators", 1, 1024),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "nthread": 5
            },
        },
    }

    logger.info("model:\n{:}".format(task["model"]))
    evals_result = dict()
    model = init_instance_by_config(task["model"])
    model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"])


if __name__ == "__main__":
    logger.info("Qlib intialization")
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri)

    logger.info("Dataset intialization")
    dataset = init_instance_by_config(dataset_config)

    logger.info("Start parameter tuning")
    study = optuna.Study(study_name="XGBoost_360", storage="sqlite:///db.sqlite3")
    study.optimize(objective)
