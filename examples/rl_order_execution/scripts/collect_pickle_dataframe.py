# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pickle
import pandas as pd
from joblib import Parallel, delayed

os.makedirs(os.path.join("data", "pickle_dataframe"), exist_ok=True)


def _collect(df: pd.DataFrame, instrument: str, tag: str) -> None:
    cur = df[df["instrument"] == instrument].sort_values(by=["datetime"])
    cur = cur.set_index(["instrument", "datetime", "date"])
    pickle.dump(cur, open(os.path.join("data", "pickle_dataframe", tag, f"{instrument}.pkl"), "wb"))


for tag in ("backtest", "feature"):
    df = pickle.load(open(os.path.join("data", "pickle", f"{tag}.pkl"), "rb"))
    df = pd.concat(list(df.values())).reset_index()
    df["date"] = df["datetime"].dt.date.astype("datetime64")
    instruments = sorted(set(df["instrument"]))

    os.makedirs(os.path.join("data", "pickle_dataframe", tag), exist_ok=True)

    Parallel(n_jobs=-1, verbose=10)(delayed(_collect)(df, instrument, tag) for instrument in instruments)
