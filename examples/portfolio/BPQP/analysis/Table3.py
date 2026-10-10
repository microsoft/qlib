from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def read_data(path, early_stop_n=1):
    import os
    import pickle
    from pathlib import Path
    path = Path(path)
    with open(os.path.join(path), "rb") as f:
        df_results = pickle.load(f)

    df_results = pd.DataFrame(df_results)

    df_results = df_results.nlargest(early_stop_n, "valid_score")  # early stop by valid_score

    focus_cols = {
        "test_ic": "IC",
        "test_rank_ic": "Rank IC",
        "test_rank_icir": "Rank ICIR",
        "test_icir": "ICIR",
        "test_mse": "MSE",
        "test_mdd": "MDD.",
        'Ann.Ret.(%)': 'Ann.Ret.(%)',
        'Sharpe': 'Sharpe',
    }

    df_results['Ann.Ret.'] = df_results['test_avg_ret'] * 240
    df_results['Ann.Vol.'] = df_results['test_avg_std'] * 240**0.5
    df_results['Sharpe'] = df_results['Ann.Ret.'] / df_results['Ann.Vol.']

    df_results['Ann.Ret.(%)'] = df_results['Ann.Ret.'] * 100
    return df_results[focus_cols.keys()].rename(columns=focus_cols.get)


def read_method(key="2ST", early_stop_n=1):
    df = []
    for exp_path in Path(DIRNAME / f'data/Table3/{key}').glob("**/?output.dict"):
        df.append(read_data(exp_path, early_stop_n=early_stop_n))
    return pd.concat({key: pd.concat(df)}, axis=0)


base_df_mlp = read_method("2ST/MLP", early_stop_n=3)
our_df_mlp = read_method("E2E/MLP", early_stop_n=3)
dc3_df_mlp = read_method("DC3/mlp", early_stop_n=3).append(read_method("DC3/dc3.old", early_stop_n=3)).append(
    read_method("DC3/mlp2023-5-18/", early_stop_n=3))
dc3_df_mlp = pd.concat({"DC3": dc3_df_mlp.droplevel(0)})

selected_cols = {
    "Prediction Metrics": ["MSE", "IC", "ICIR"],
    "Portfolio Metrics": ["Ann.Ret.(%)", "Sharpe"],
}


def add_header(df):
    return pd.concat({k: df.loc[:, cols] for k, cols in selected_cols.items()}, axis=1)


def rename(x):
    d = {
        "MLP": "Two-Stage",
        "MLP+BPQP": "COFFEE",
        "MLP+naïve NN": "naïve NN",
        "MLP+DC3": "DC3",
        "Predictive Metrics": "Prediction Metrics",
        # New:
        "2ST": "Two-Stage",
        "2ST/MLP": "Two-Stage",
        "E2E/MLP": "BPQP"
    }
    return d.get(x, x)


def get_key(x):
    if isinstance(x, pd.Index):
        x = x.to_series()
    x = x.apply(
        lambda i: "Two-Stage	naïve NN	DC3	BPQP	COFFEE	MSE	IC	ICIR	Ann.Ret.	Sharpe Prediction Metrics	Portfolio Metrics".
        find(i))
    return x


df_all = pd.concat([add_header(base_df_mlp), add_header(our_df_mlp), add_header(dc3_df_mlp)])

df_all.index.names = ['method', 'runs']

from functools import partial


def get_prec(s):
    prec = 1 - np.floor(np.log10(s.abs().min()))
    return prec


def merge_runs(s, prec):
    ft = "%%.%df" % prec
    # print(("%s(±%s)" % (ft, ft)))
    return ("%s(±%s)" % (ft, ft)) % (s.mean(), s.std())


df_all = df_all.apply(lambda s: s.groupby('method').apply(partial(merge_runs, prec=get_prec(s))))
vis_df = df_all.rename(rename).sort_index().sort_index(key=get_key, axis=1).sort_index(key=get_key)
print(vis_df.to_latex())
