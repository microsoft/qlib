# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import platform
import shutil
import sys
import tempfile
from importlib import import_module

import yaml


DELETE_KEY = "_delete_"


def merge_a_into_b(a: dict, b: dict) -> dict:
    b = b.copy()
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            v.pop(DELETE_KEY, False)
            b[k] = merge_a_into_b(v, b[k])
        else:
            b[k] = v
    return b


def check_file_exist(filename: str, msg_tmpl: str = 'file "{}" does not exist') -> None:
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def parse_backtest_config(path: str) -> dict:
    abs_path = os.path.abspath(path)
    check_file_exist(abs_path)

    file_ext_name = os.path.splitext(abs_path)[1]
    if file_ext_name not in (".py", ".json", ".yaml", ".yml"):
        raise IOError("Only py/yml/yaml/json type are supported now!")

    with tempfile.TemporaryDirectory() as tmp_config_dir:
        with tempfile.NamedTemporaryFile(dir=tmp_config_dir, suffix=file_ext_name) as tmp_config_file:
            if platform.system() == "Windows":
                tmp_config_file.close()

            tmp_config_name = os.path.basename(tmp_config_file.name)
            shutil.copyfile(abs_path, tmp_config_file.name)

            if abs_path.endswith(".py"):
                tmp_module_name = os.path.splitext(tmp_config_name)[0]
                sys.path.insert(0, tmp_config_dir)
                module = import_module(tmp_module_name)
                sys.path.pop(0)

                config = {k: v for k, v in module.__dict__.items() if not k.startswith("__")}

                del sys.modules[tmp_module_name]
            else:
                with open(tmp_config_file.name) as input_stream:
                    config = yaml.safe_load(input_stream)

    if "_base_" in config:
        base_file_name = config.pop("_base_")
        if not isinstance(base_file_name, list):
            base_file_name = [base_file_name]

        for f in base_file_name:
            base_config = parse_backtest_config(os.path.join(os.path.dirname(abs_path), f))
            config = merge_a_into_b(a=config, b=base_config)

    return config


def _convert_all_list_to_tuple(config: dict) -> dict:
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = tuple(v)
        elif isinstance(v, dict):
            config[k] = _convert_all_list_to_tuple(v)
    return config


def get_backtest_config_fromfile(path: str) -> dict:
    backtest_config = parse_backtest_config(path)

    exchange_config_default = {
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5.0,
        "trade_unit": 100.0,
        "cash_limit": None,
    }
    backtest_config["exchange"] = merge_a_into_b(a=backtest_config["exchange"], b=exchange_config_default)
    backtest_config["exchange"] = _convert_all_list_to_tuple(backtest_config["exchange"])

    backtest_config_default = {
        "debug_single_stock": None,
        "debug_single_day": None,
        "concurrency": -1,
        "multiplier": 1.0,
        "output_dir": "outputs/",
        "generate_report": False,
    }
    backtest_config = merge_a_into_b(a=backtest_config, b=backtest_config_default)

    return backtest_config
