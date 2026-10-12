# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import shutil
from copy import deepcopy
from pathlib import Path

from ruamel.yaml import YAML

from qlib.contrib.data.highfreq_provider import HighFreqProvider


def load_script_config(config_path: str) -> dict:
    """Load an RL data-gen YAML config with the safe ruamel loader.

    Rejects ``yaml.FullLoader`` semantics: only plain mappings/lists/scalars
    are accepted, so untrusted config files cannot construct arbitrary objects.
    """
    yaml = YAML(typ="safe", pure=True)
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"RL data-gen config must be a mapping: {config_path}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yml")
    parser.add_argument("-d", "--dest", type=str, default=".")
    parser.add_argument("-s", "--split", type=str, choices=["none", "date", "stock", "both"], default="stock")
    args = parser.parse_args()

    conf = load_script_config(args.config)

    for k, v in conf.items():
        if isinstance(v, dict) and "path" in v:
            v["path"] = os.path.join(args.dest, v["path"])
    provider = HighFreqProvider(**conf)

    # Gen dataframe
    if "feature_conf" in conf:
        feature = provider._gen_dataframe(deepcopy(provider.feature_conf))
    if "backtest_conf" in conf:
        backtest = provider._gen_dataframe(deepcopy(provider.backtest_conf))

    provider.feature_conf["path"] = os.path.splitext(provider.feature_conf["path"])[0] + "/"
    provider.backtest_conf["path"] = os.path.splitext(provider.backtest_conf["path"])[0] + "/"
    # Split by date
    if args.split == "date" or args.split == "both":
        provider._gen_day_dataset(deepcopy(provider.feature_conf), "feature")
        provider._gen_day_dataset(deepcopy(provider.backtest_conf), "backtest")

    # Split by stock
    if args.split == "stock" or args.split == "both":
        provider._gen_stock_dataset(deepcopy(provider.feature_conf), "feature")
        provider._gen_stock_dataset(deepcopy(provider.backtest_conf), "backtest")

    shutil.rmtree("stat/", ignore_errors=True)
