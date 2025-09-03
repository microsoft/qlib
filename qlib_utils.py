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
import sys
import subprocess
import copy

# --- Factor and Model Configurations ---
HANDLER_ALPHA158 = { "class": "Alpha158", "module_path": "qlib.contrib.data.handler", "kwargs": { "start_time": "2014-01-01", "end_time": "2022-12-31", "fit_start_time": "2014-01-01", "fit_end_time": "2019-12-31", "instruments": "csi300" } }
HANDLER_ALPHA360 = { "class": "Alpha360", "module_path": "qlib.contrib.data.handler", "kwargs": { "start_time": "2014-01-01", "end_time": "2022-12-31", "fit_start_time": "2014-01-01", "fit_end_time": "2019-12-31", "instruments": "csi300" } }
DATASET_ALPHA158 = { "class": "DatasetH", "module_path": "qlib.data.dataset", "kwargs": { "handler": HANDLER_ALPHA158, "segments": { "train": ("2014-01-01", "2019-12-31"), "valid": ("2020-01-01", "2020-12-31"), "test": ("2021-01-01", "2022-12-31") } } }
DATASET_ALPHA360 = { "class": "DatasetH", "module_path": "qlib.data.dataset", "kwargs": { "handler": HANDLER_ALPHA360, "segments": { "train": ("2014-01-01", "2019-12-31"), "valid": ("2020-01-01", "2020-12-31"), "test": ("2021-01-01", "2022-12-31") } } }
LIGHTGBM_MODEL = { "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt", "kwargs": { "loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421, "subsample": 0.8789, "n_estimators": 200, "max_depth": 8 } }
XGBOOST_MODEL = { "class": "XGBModel", "module_path": "qlib.contrib.model.xgboost", "kwargs": { "n_estimators": 200, "learning_rate": 0.05, "max_depth": 7 } }
CATBOOST_MODEL = { "class": "CatBoostModel", "module_path": "qlib.contrib.model.catboost", "kwargs": { "iterations": 200, "learning_rate": 0.05, "depth": 7 } }
ALSTM_MODEL = { "class": "ALSTM", "module_path": "qlib.contrib.model.pytorch_alstm_ts", "kwargs": { "d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.5, "n_epochs": 30, "lr": 1e-4, "early_stop": 5 } }
SUPPORTED_MODELS = {
    "LightGBM (Alpha158)": {"task": {"model": LIGHTGBM_MODEL, "dataset": DATASET_ALPHA158}},
    "LightGBM (Alpha360)": {"task": {"model": LIGHTGBM_MODEL, "dataset": DATASET_ALPHA360}},
    "XGBoost (Alpha158)": {"task": {"model": XGBOOST_MODEL, "dataset": DATASET_ALPHA158}},
    "CatBoost (Alpha158)": {"task": {"model": CATBOOST_MODEL, "dataset": DATASET_ALPHA158}},
    "ALSTM (Alpha158)": {"task": {"model": ALSTM_MODEL, "dataset": DATASET_ALPHA158}},
}

# --- Data Management Functions ---
def get_script_path(script_name):
    app_dir = Path().resolve()
    script_path = app_dir / "scripts" / script_name
    if not script_path.exists():
        script_path = app_dir / "scripts" / "data_collector" / "yahoo" / script_name
        if not script_path.exists():
             raise FileNotFoundError(f"Script '{script_name}' not found. Please ensure the 'scripts' folder from the Qlib GitHub repository is in the same directory as the application.")
    return str(script_path)

def run_command_with_log(command, log_placeholder):
    log_text = f"Running command: {command}\n\n"
    log_placeholder.code(log_text, language='log')
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    for line in iter(process.stdout.readline, ''):
        log_text += line
        log_placeholder.code(log_text, language='log')
    process.stdout.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output=log_text)

def update_daily_data(qlib_dir, start_date, end_date, log_placeholder):
    script_path = get_script_path("collector.py")
    command = f'"{sys.executable}" "{script_path}" update_data_to_bin --qlib_data_1d_dir "{qlib_dir}" --trading_date {start_date} --end_date {end_date}'
    run_command_with_log(command, log_placeholder)

def check_data_health(qlib_dir, log_placeholder):
    script_path = get_script_path("check_data_health.py")
    command = f'"{sys.executable}" "{script_path}" check_data --qlib_dir "{qlib_dir}"'
    run_command_with_log(command, log_placeholder)

# --- Model Training & Evaluation Functions (FIXED) ---
def train_model(model_name: str, qlib_dir: str, models_save_dir: str, custom_config: dict = None, custom_model_name: str = None, stock_pool: str = 'csi300', finetune_model_path: str = None):
    provider_uri = str(Path(qlib_dir).expanduser())
    if not exists_qlib_data(provider_uri):
        raise FileNotFoundError("Qlib data not found.")
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    model_config = copy.deepcopy(custom_config if custom_config is not None else SUPPORTED_MODELS[model_name])
    model_config["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = stock_pool
    dataset = init_instance_by_config(model_config["task"]["dataset"])

    model_kwargs = model_config["task"]["model"]["kwargs"]
    if finetune_model_path:
        with open(finetune_model_path, 'rb') as f:
            initial_model = pickle.load(f)
        model_kwargs['init_model'] = initial_model

    model = init_instance_by_config(model_config["task"]["model"])
    model.fit(dataset)

    model_basename = custom_model_name if custom_model_name else model_name.replace(' ', '_').replace('(', '').replace(')', '')
    model_save_path = Path(models_save_dir).expanduser() / f"{model_basename}.pkl"
    config_save_path = Path(models_save_dir).expanduser() / f"{model_basename}.yaml"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    model_config["task"]["model"]["kwargs"].pop('init_model', None)

    with open(model_save_path, 'wb') as f: pickle.dump(model, f)
    with open(config_save_path, 'w') as f: yaml.dump(model_config["task"], f)
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
        raise FileNotFoundError("Qlib data not found.")
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
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
        raise FileNotFoundError("Qlib data not found.")
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    config["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = start_time
    config["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = end_time
    dataset = init_instance_by_config(config["dataset"])
    strategy = TopkDropoutStrategy(model=model, dataset=dataset, **strategy_kwargs)
    report_df, _ = backtest_daily(start_time=start_time, end_time=end_time, strategy=strategy, exchange_kwargs=exchange_kwargs)
    return report_df
