import pickle
import os
import pandas as pd
from tqdm import tqdm

for tag in ["test", "valid"]:
    files = os.listdir(os.path.join("data/orders/", tag))
    dfs = []
    for f in tqdm(files):
        df = pickle.load(open(os.path.join("data/orders/", tag, f), "rb"))
        df = df.drop(["$close0"], axis=1)
        dfs.append(df)

    total_df = pd.concat(dfs)
    pickle.dump(total_df, open(os.path.join("data", "orders", f"{tag}_orders.pkl"), "wb"))
