import qlib
from qlib.config import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
import optuna

provider_uri = "~/.qlib/qlib_data/cn_data"
if not exists_qlib_data(provider_uri):
    print(f"Qlib data is not found in {provider_uri}")
    sys.path.append(str(scripts_dir))
    from get_data import GetData

    GetData().qlib_data(target_dir=provider_uri, region="cn")
qlib.init(provider_uri=provider_uri, region="cn")

market = "csi300"
benchmark = "SH000300"

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
}
dataset_task = {
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
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
dataset = init_instance_by_config(dataset_task["dataset"])


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

    evals_result = dict()
    model = init_instance_by_config(task["model"])
    model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"])


study = optuna.Study(study_name="LGBM_158", storage="sqlite:///db.sqlite3")
study.optimize(objective, n_jobs=6)
