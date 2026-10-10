import numpy as np
import pandas as pd
from pathlib import Path
import re

DIRNAME = Path(__file__).absolute().resolve().parent
OURM = "OSQP"

# # Outlines: Loading Figure1 data
def load_lp_qp_data(fname):

    data_path = DIRNAME / "data/Table1"

    df = pd.read_csv(data_path / fname, index_col=0)

    df = df.loc[:, df.columns.str.lower().str.startswith("time")]

    # Add only forward
    for col in df.columns:
        if OURM in col and "Forward" in col:
            fcol = col.replace(OURM, f"{OURM}[O.F.]")
            df[fcol] = df[col]
            bcol = fcol.replace("Forward", "Backward")
            df[bcol] = df[col.replace(OURM, "Exact").replace("Forward", "Backward")]

    # Add All time col
    for col in df.columns:
        if "Forward" in col:
            back_col = col.replace("Forward", "Backward")
            df[col.replace("Forward", "All")] = df[col] + df[back_col]

    def format_df(df):
        df = df.to_frame("time")
        from collections import defaultdict

        new_col = defaultdict(list)

        for row in df.iterrows():
            m = re.match(
                r"Time (?P<method>(\w|\[O\.F\.\])+) (?P<pass>\w+) ndim:(?P<var_n>\d+) neq=nineq=(?P<con_n>\d+)", row[0])
            gd = m.groupdict()
            gd['size'] = f"{gd['var_n']}x{gd['con_n']}"
            for k, v in gd.items():
                new_col[k].append(v)

        for col, values in new_col.items():
            df[col] = values

        df['pass'] = df['pass'].apply(lambda x: f"{x[0]}.")

        df = df.set_index(["method", "pass", "var_n", "con_n", "size"])

        df = df.iloc[:, 0].unstack('pass')
        return df

    df_mean, df_std = format_df(df.mean()), format_df(df.std())

    return df_mean, df_std


# %% [markdown]
# ## Outlines: SOCP data
# DATA_FOLDER = "SOCP"
DATA_FOLDER = "SOCPV02"


## loading
def format_data(df, method_name):
    name_map = {
        "10": (10, 5, "10x5"),
        "50": (50, 10, "50x10"),
        "100": (100, 20, "100x20"),
        "500": (500, 100, "500x100"),
    }
    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(((ps, *name_map[var_n]) for ps, var_n in df.columns))
    df = pd.concat({method_name: df}, axis=1)
    df.columns.names = "method	pass var_n	con_n	size".split()
    return df


def load_socp_data():
    folder = DIRNAME / "data" / "Table1" / DATA_FOLDER
    # SCS
    df_b = pd.read_csv(folder / "cvxpy_backward_time.csv", index_col=0)  # TODO
    df_f = pd.read_csv(folder / "cvxpy_forward_time.csv", index_col=0)  # TODO

    # BPQP ECOS
    df_b_o = pd.read_csv(folder / "bpqp_backward_time.csv", index_col=0)  # TODO
    df_f_o = pd.read_csv(folder / "ecos_forward.csv", index_col=0)  # TODO

    # Exact
    df_b_ex = pd.read_csv(folder / "exact_back_time.csv", index_col=0)  # TODO

    # BPQP Only Ours
    df_cvxpy = format_data(pd.concat({'B.': df_b, "F.": df_f, "A.": df_f + df_b}, axis=1), "CVXPY")
    df_bpqp = format_data(pd.concat({'B.': df_b_o, "F.": df_f_o, "A.": df_f_o + df_b_o}, axis=1), "OSQP")
    df_ex = format_data(pd.concat({'B.': df_b_ex, "F.": df_b_ex.head(0), "A.": df_b_ex.head(0)}, axis=1), "Exact")
    df_of = format_data(pd.concat({'B.': df_b_ex, "F.": df_f_o, "A.": df_f_o + df_b_ex}, axis=1), "OSQP[O.F.]")

    df_all =  pd.concat([df_bpqp, df_cvxpy, df_ex, df_of], axis=1)
    df_mean = df_all.mean().unstack('pass')
    df_std = df_all.std().unstack('pass')
    return df_mean, df_std
