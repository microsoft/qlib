
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
            "class": "CatBoostModel",
            "module_path": "qlib.contrib.model.catboost_model",
            "kwargs": {
                "loss": "RMSE",
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e0),
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "subsample": trial.suggest_float("subsample", 0.1, 1),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "thread_count": 20,
                "bootstrap_type": "Poisson",
                "grow_policy": "Lossguide"
            },
        }
      }

    logger.info("model:\n{:}".format(task["model"]))
    evals_result = dict()
    model = init_instance_by_config(task["model"])
    evals_result = model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"])


if __name__ == "__main__":
    logger.info("Qlib intialization")
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri)

    logger.info("Dataset intialization")
    dataset = init_instance_by_config(dataset_config)

    logger.info("Start parameter tuning")
    study = optuna.Study(study_name="CatBoost_360", storage="sqlite:///db.sqlite3")
    study.optimize(objective)
