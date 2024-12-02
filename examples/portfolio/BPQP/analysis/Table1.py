from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from data_loader import OURM, load_lp_qp_data, load_socp_data

DIRNAME = Path(__file__).absolute().resolve().parent

sys.path.append(str(DIRNAME))

method_order = [OURM, f"{OURM}+Exact", f"{OURM}[O.F.]", "DC3", "Qpth", "QPTH", 'CVXPY', 'Exact']
pass_order = ['F.', "B.", 'A.']
ppname = "BPQP"


def print_latex_table(df_mean, df_std, method_grp):
    print("=" * 50, method_grp, "=" * 50)

    def get_key(x):
        if hasattr(x, "to_series"):  # incase of index
            x = x.to_series()

        cat_list = method_order + pass_order
        try:
            x = x.apply(cat_list.index)
            return x
        except ValueError:
            pass
        m = re.match(r"\d+(x|\*|×)\d+", x.iloc[0])
        if m is not None:
            sep = m.groups()[0]
            x = x.apply(lambda z: int(z.split(sep)[0]))
        return x

    def rename(x):
        # print(x)
        if OURM in x:
            return x.replace(OURM, ppname)
        mp = {
            "Qpth": "qpth/OptNet",
            "QPTH": "qpth/OptNet",
            "F.": "Forward",
            "B.": "Backward",
            "A.": "Total(Forward + Backward)",
        }
        return mp.get(x, x)

    def tex_rename(x):
        if 'x' in x:
            return x.replace("x", r"×")
        return x

    def get_tex_df(df):
        text_df2 = df
        text_df2 = text_df2.stack(dropna=False).to_frame("time").reset_index()
        text_df2 = text_df2.sort_values(['size', 'method', 'pass'], key=get_key)
        tex_df = text_df2.set_index(['pass', 'method',
                                     'size'])['time'].unstack('size').stack().unstack('pass').unstack().sort_index(
                                         key=get_key).sort_index(axis=1, key=get_key).iloc[
                                             ::-1,
                                         ]
        tex_df = tex_df.rename(rename, axis=0).rename(rename, axis=1)
        return tex_df

    def add_level(df, name, idx_name):
        names = [idx_name] + list(df.index.names)
        df = pd.concat({name: df})
        df.index.names = names
        return df

    tex_df = get_tex_df(df_mean)
    tex_df_std = get_tex_df(df_std)

    scale = 10**np.ceil(np.log10(tex_df.min().min()))

    def cbf(s1, s2):
        if np.isnan(s1):
            return "-"
        return "{:.1f}(±{:.1f})".format(s1, s2)

    def cbf2(s1, s2):
        return s1.combine(s2, cbf)

    tex_df_all = (tex_df / scale).combine((tex_df_std / scale), cbf2)

    tex_df_all = tex_df_all.reindex(tex_df.index, axis=0).reindex(tex_df.columns, axis=1)

    print("(scale {:.1e})".format(scale))

    print(
        add_level(add_level(tex_df_all.drop("Forward", axis=1, level=0), 'abs. time', 'metric'), method_grp,
                  'dataset').rename(columns=tex_rename).to_latex())


# QP or LP data
for fname in [
        "BPQP_QP_raw_results.csv",  # selected QP results
        "BPQP_LP_raw_results.csv",  # Selected LP
]:
    method_grp = fname[5:7]
    df_mean, df_std = load_lp_qp_data(fname)
    df_std = df_std.reindex(df_mean.index)
    print_latex_table(df_mean, df_std, method_grp)

# ## SOCP loader
df_mean, df_std = load_socp_data()
print_latex_table(df_mean, df_std, "SOCP")
