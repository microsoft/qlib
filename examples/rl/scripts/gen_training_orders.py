# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import pandas as pd
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=20220926)
parser.add_argument("--stock", type=str, default="AAPL")
parser.add_argument("--train_size", type=int, default=10)
parser.add_argument("--valid_size", type=int, default=2)
parser.add_argument("--test_size", type=int, default=2)
args = parser.parse_args()

np.random.seed(args.seed)

os.makedirs(os.path.join("data", "training_order_split"), exist_ok=True)

for group, n in zip(("train", "valid", "test"), (args.train_size, args.valid_size, args.test_size)):
    path = os.path.join("data", "pickle", f"backtest{group}.pkl")
    df = pickle.load(open(path, "rb")).reset_index()
    df["date"] = df["datetime"].dt.date.astype("datetime64")

    dates = sorted(set([str(d).split(" ")[0] for d in df["date"]]))

    data_df = pd.DataFrame(
        {
            "date": sorted(np.random.choice(dates, size=n, replace=False)),
            "instrument": [args.stock] * n,
            "amount": np.random.randint(low=3, high=11, size=n) * 100.0,
            "order_type": [0] * n,
        }
    ).set_index(["date", "instrument"])

    os.makedirs(os.path.join("data", "training_order_split", group), exist_ok=True)
    pickle.dump(data_df, open(os.path.join("data", "training_order_split", group, f"{args.stock}.pkl"), "wb"))
