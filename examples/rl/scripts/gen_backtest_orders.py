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

path = os.path.join("data", "pickle", "backtesttest.pkl")
df = pickle.load(open(path, "rb")).reset_index()
df["date"] = df["datetime"].dt.date.astype("datetime64")

instruments = sorted(set(df["instrument"]))

# TODO: The example is expected to be able to handle data containing missing values.
# TODO: Currently, we just simply skip dates that contain missing data. We will add
# TODO: this feature in the future.
skip_dates = {}
for instrument in instruments:
    csv_df = pd.read_csv(os.path.join("data", "csv", f"{instrument}.csv"))
    csv_df = csv_df[csv_df["close"].isna()]
    dates = set([str(d).split(" ")[0] for d in csv_df["date"]])
    skip_dates[instrument] = dates

df_list = []
for instrument in instruments:
    print(instrument)

    cur_df = df[df["instrument"] == instrument]

    dates = sorted(set([str(d).split(" ")[0] for d in cur_df["date"]]))
    dates = [date for date in dates if date not in skip_dates[instrument]]

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
