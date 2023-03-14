# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

DATA_PATH = Path(os.path.join("data", "pickle_dataframe", "backtest"))
OUTPUT_PATH = Path(os.path.join("data", "orders"))


def generate_order(stock: str, start_idx: int, end_idx: int) -> None:
    df = pd.read_pickle(DATA_PATH / f"{stock}.pkl")
    df = df.groupby("date").take(range(start_idx, end_idx)).droplevel(level=0)
    div = df["$volume0"].rolling((end_idx - start_idx) * 60).mean().shift(1).groupby(level="date").transform("first")

    order_all = pd.DataFrame(df.groupby(level=(2, 0)).mean().dropna())
    order_all["amount"] = np.random.lognormal(-3.28, 1.14) * order_all["$volume0"]
    order_all = order_all[order_all["amount"] > 0.0]
    order_all["order_type"] = 0
    order_all = order_all.drop(columns=["$volume0"])

    order_train = order_all[order_all.index.get_level_values(0) <= pd.Timestamp("2021-06-30")]
    order_test = order_all[order_all.index.get_level_values(0) > pd.Timestamp("2021-06-30")]
    order_valid = order_test[order_test.index.get_level_values(0) <= pd.Timestamp("2021-09-30")]
    order_test = order_test[order_test.index.get_level_values(0) > pd.Timestamp("2021-09-30")]

    for order, tag in zip((order_train, order_valid, order_test, order_all), ("train", "valid", "test", "all")):
        path = OUTPUT_PATH / tag
        os.makedirs(path, exist_ok=True)
        if len(order) > 0:
            order.to_pickle(path / f"{stock}.pkl.target")


np.random.seed(1234)
file_list = sorted(os.listdir(DATA_PATH))
stocks = [f.replace(".pkl", "") for f in file_list]
stocks = sorted(np.random.choice(stocks, size=100, replace=False))
for stock in tqdm(stocks):
    generate_order(stock, 0, 240 // 5 - 1)
