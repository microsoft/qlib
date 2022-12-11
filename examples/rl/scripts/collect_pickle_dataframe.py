# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pickle
import pandas as pd
from tqdm import tqdm

os.makedirs(os.path.join("data", "pickle_dataframe"), exist_ok=True)

for tag in ("backtest", "feature"):
    df = pickle.load(open(os.path.join("data", "pickle", f"{tag}.pkl"), "rb"))
    df = pd.concat(list(df.values())).reset_index()
    df["date"] = df["datetime"].dt.date.astype("datetime64")
    instruments = sorted(set(df["instrument"]))

    os.makedirs(os.path.join("data", "pickle_dataframe", tag), exist_ok=True)
    for instrument in tqdm(instruments):
        cur = df[df["instrument"] == instrument].sort_values(by=["datetime"])
        cur = cur.set_index(["instrument", "datetime", "date"])
        pickle.dump(cur, open(os.path.join("data", "pickle_dataframe", tag, f"{instrument}.pkl"), "wb"))
