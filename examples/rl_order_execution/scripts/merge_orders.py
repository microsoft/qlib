import os
import pandas as pd
from tqdm import tqdm

from qlib.utils.pickle_utils import restricted_pickle_load

for tag in ["test", "valid"]:
    files = os.listdir(os.path.join("data/orders/", tag))
    dfs = []
    for f in tqdm(files):
        with open(os.path.join("data/orders/", tag, f), "rb") as fr:
            df = restricted_pickle_load(fr)
        df = df.drop(["$close0"], axis=1)
        dfs.append(df)

    total_df = pd.concat(dfs)
    pickle.dump(total_df, open(os.path.join("data", "orders", f"{tag}_orders.pkl"), "wb"))
