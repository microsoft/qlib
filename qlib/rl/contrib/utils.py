# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_order_file(order_file: Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(order_file, pd.DataFrame):
        return order_file

    order_file = Path(order_file)

    if order_file.suffix == ".pkl":
        order_df = pd.read_pickle(order_file).reset_index()
    elif order_file.suffix == ".csv":
        order_df = pd.read_csv(order_file)
    else:
        raise TypeError(f"Unsupported order file type: {order_file}")

    if "date" in order_df.columns:
        # legacy dataframe columns
        order_df = order_df.rename(columns={"date": "datetime", "order_type": "direction"})
    order_df["datetime"] = order_df["datetime"].astype(str)

    return order_df
