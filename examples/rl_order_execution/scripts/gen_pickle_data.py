# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import yaml
import argparse
import os
import shutil
from copy import deepcopy

from qlib.contrib.data.highfreq_provider import HighFreqProvider

loader = yaml.FullLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yml")
    parser.add_argument("-d", "--dest", type=str, default=".")
    parser.add_argument("-s", "--split", type=str, choices=["none", "date", "stock", "both"], default="stock")
    args = parser.parse_args()

    conf = yaml.load(open(args.config), Loader=loader)

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
