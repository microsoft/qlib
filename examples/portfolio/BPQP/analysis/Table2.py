from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent

data_path = DIRNAME / "data" / "Table2"

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from e2eutils import rename, add_level, get_key, tex_rename

import re


def load_qp_acc():

    acc = pd.read_csv(data_path / 'BPQP_QP_results.csv')

    acc['Err.'] = acc.loc[:, "0"].apply(lambda x: re.match(r"(.*)\((.*)\)", x).groups()[0])
    acc['Std.'] = acc.loc[:, "0"].apply(lambda x: re.match(r"(.*)\((.*)\)", x).groups()[1])

    acc = acc.set_index('avg')

    acc = acc.loc[acc.index.str.startswith("Acc"), :]

    acc = acc.loc[~acc.index.str.startswith("Accuracy Forward"), :]

    from collections import defaultdict

    new_col = defaultdict(list)

    for row in acc.iterrows():
        m = re.match(r"Accuracy (?P<method>\w+) (?P<pass>\w+) ndim:(?P<var_n>\d+) neq=nineq=(?P<con_n>\d+)", row[0])
        gd = m.groupdict()
        gd['scale'] = f"{gd['var_n']}x{gd['con_n']}"
        for k, v in gd.items():
            new_col[k].append(v)

    for col, values in new_col.items():
        acc[col] = values

    acc = acc.loc[:, ['scale', 'method', 'pass', 'Err.', 'Std.']]

    df = acc.set_index([
        'scale',
        'method',
        'pass',
    ]).loc[:, 'Err.'].unstack('scale').swaplevel()
    df_std = acc.set_index([
        'scale',
        'method',
        'pass',
    ]).loc[:, 'Std.'].unstack('scale').swaplevel()

    # final_df = acc.rename(tex_rename, axis=1).append(dc3.swaplevel().rename(tex_rename, axis=1))
    final_df = df.rename(tex_rename, axis=1)
    final_df_std = df_std.rename(tex_rename, axis=1)

    final_df = final_df.rename({"Backward": "QP"})

    final_df_std = final_df_std.rename({"Backward": "QP"})
    return final_df, final_df_std


final_df, final_df_std = load_qp_acc()

# load ospq
df = pd.concat({
    "CVXPY": pd.read_csv(data_path / 'cp_acc.csv', index_col=0),  #.iloc[50:],
    "OSQP": pd.read_csv(data_path / 'bpqp_acc.csv', index_col=0),  # .iloc[:50],
}, axis=1).rename(columns=rename)  # .unstack(level=0)

final_df = final_df.append(pd.concat({
    "SOCP": df.mean().unstack(),
}, axis=0))

final_df_std = final_df_std.append(pd.concat({
    "SOCP": df.std().unstack(),
}, axis=0))


def show_table(final_df, final_df_std):

    def agg_info(final_df):
        final_df = final_df.astype("float")
        final_df = final_df.sort_index(key=get_key).sort_index(key=get_key, axis=1)
        final_df = final_df.rename(rename, axis=0)
        final_df.columns.name = 'scale'
        final_df = final_df.iloc[
            ::-1,
        ]

        final_df = final_df.dropna(axis=1)

        from scipy.stats import gmean

        final_df = final_df.apply(gmean, axis=1).to_frame("Avg. Err.")
        return final_df

    df = agg_info(final_df)

    # print(final_df.T.to_latex( float_format="{:.1e}".format, na_rep="-"))

    df_std = agg_info(final_df_std)

    def cbf(s1, s2):
        if np.isnan(s1):
            return "-"
        return "{:.2e}(±{:.2e})".format(s1, s2)

    def cbf2(s1, s2):
        return s1.combine(s2, cbf)

    df_all = df.combine(df_std, cbf2)

    df_all = df_all.T.sort_index(key=get_key, axis=1)
    df_all.columns.names = ['', "method"]

    print(df_all.to_latex())


show_table(final_df, final_df_std)
