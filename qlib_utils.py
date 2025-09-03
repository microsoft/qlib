import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.strategy import TopkDropoutStrategy
from pathlib import Path
import yaml
import streamlit as st
import pandas as pd
import pickle

# LightGBM Model Config
LIGHTGBM_CONFIG = {
    "task": {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "n_estimators": 200,
                "max_depth": 8,
                "num_leaves": 210,
                "min_child_samples": 5,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "start_time": "2014-01-01",
                        "end_time": "2020-12-31",
                        "fit_start_time": "2014-01-01",
                        "fit_end_time": "2018-12-31",
                        "instruments": "csi300",
                    },
                },
                "segments": {
                    "train": ("2014-01-01", "2018-12-31"),
                    "valid": ("2019-01-01", "2019-12-31"),
                    "test": ("2020-01-01", "2020-12-31"),
                },
            },
        },
    },
}

XGBOOST_CONFIG = {
    "task": {
        "model": {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
        "dataset": LIGHTGBM_CONFIG["task"]["dataset"],  # Reuse the same dataset config
    },
}

SUPPORTED_MODELS = {
    "LightGBM (Alpha158)": LIGHTGBM_CONFIG,
    "XGBoost (Alpha158)": XGBOOST_CONFIG,
}

def train_model(model_name: str, qlib_dir: str, models_save_dir: str, custom_config: dict = None):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_name} is not supported.")

    provider_uri = str(Path(qlib_dir).expanduser())
    if not exists_qlib_data(provider_uri):
        raise FileNotFoundError("Qlib data not found. Please download the data first.")

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # Use the user's custom config if provided, otherwise use the default
    model_config = custom_config if custom_config is not None else SUPPORTED_MODELS[model_name]

    dataset = init_instance_by_config(model_config["task"]["dataset"])
    model = init_instance_by_config(model_config["task"]["model"])

    with R.start(experiment_name="streamlit_train_custom"):
        model.fit(dataset)

        model_basename = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        model_save_path = Path(models_save_dir).expanduser() / f"{model_basename}.pkl"
        config_save_path = Path(models_save_dir).expanduser() / f"{model_basename}.yaml"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model with pickle and config with yaml
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)
        with open(config_save_path, 'w') as f:
            yaml.dump(model_config["task"], f)

        return str(model_save_path)

def predict(model_path_str: str, qlib_dir: str, prediction_date: str):
    model_path = Path(model_path_str)
    config_path = model_path.with_suffix(".yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found for model {model_path.name}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    provider_uri = str(Path(qlib_dir).expanduser())
    if not exists_qlib_data(provider_uri):
        raise FileNotFoundError("Qlib data not found. Please download the data first.")

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Update dataset config for prediction
    config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = pd.to_datetime(prediction_date) - pd.DateOffset(years=2)
    config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = prediction_date
    config["dataset"]["kwargs"]["segments"] = {"predict": (prediction_date, prediction_date)}

    dataset = init_instance_by_config(config["dataset"])
    prediction = model.predict(dataset)

    prediction = prediction.reset_index().rename(columns={'instrument': 'StockID', 'datetime': 'Date'})
    return prediction.sort_values(by="score", ascending=False)

def backtest_strategy(model_path_str: str, qlib_dir: str, start_time: str, end_time: str, strategy_kwargs: dict, exchange_kwargs: dict):
    model_path = Path(model_path_str)
    config_path = model_path.with_suffix(".yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found for model {model_path.name}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    provider_uri = str(Path(qlib_dir).expanduser())
    if not exists_qlib_data(provider_uri):
        raise FileNotFoundError("Qlib data not found. Please download the data first.")

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Update dataset handler times for the backtest period
    config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = start_time
    config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = end_time

    dataset = init_instance_by_config(config["dataset"])

    strategy = TopkDropoutStrategy(
        model=model,
        dataset=dataset,
        **strategy_kwargs
    )

    report_df, _ = backtest_daily(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        exchange_kwargs=exchange_kwargs
    )

    return report_df
