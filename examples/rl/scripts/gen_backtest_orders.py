# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import pandas as pd
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=20220926)
parser.add_argument("--num_order", type=int, default=10)
args = parser.parse_args()

np.random.seed(args.seed)

path = os.path.join("data", "pickle", "backtesttest.pkl")  # TODO: rename file
df = pickle.load(open(path, "rb")).reset_index()
df["date"] = df["datetime"].dt.date.astype("datetime64")

instruments = sorted(set(df["instrument"]))
df_list = []
for instrument in instruments:
    print(instrument)

    cur_df = df[df["instrument"] == instrument]

    dates = sorted(set([str(d).split(" ")[0] for d in cur_df["date"]]))

    n = args.num_order
    df_list.append(
        pd.DataFrame(
            {
                "date": sorted(np.random.choice(dates, size=n, replace=False)),
                "instrument": [instrument] * n,
                "amount": np.random.randint(low=3, high=11, size=n) * 100.0,
                "order_type": np.random.randint(low=0, high=2, size=n),
            }
        ).set_index(["date", "instrument"]),
    )

total_df = pd.concat(df_list)
total_df.to_csv("data/backtest_orders.csv")
