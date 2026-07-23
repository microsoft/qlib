import pandas as pd
import re

ourm = "OSQP"
method_order = [ourm, f"{ourm}+Exact", f"{ourm}[O.F.]", "DC3", "Qpth","QPTH", "qpth/OptNet", 'CVXPY', 'Exact']
pass_order = ['F.', "B.", 'A.']

PPNAME = "BPQP"


def rename(x):
    # print(x)
    if ourm in x:
        return x.replace(ourm, PPNAME)
    mp = {
        "Qpth": "qpth/OptNet",
        "QPTH": "qpth/OptNet",
        "F.": "Forward",
        "B.": "Backward",
        "A.": "Total(Forward + Backward)",
        # columns
        "10": "10×5",
        "50": "50×10",
        "100": "100×20",
        "500": "500×100",
    }
    return mp.get(x, x)



def get_key(x):
    if hasattr(x, "to_series"): # incase of index
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

def tex_rename(x):
    m = re.match(r"\d+(x|\*|×)\d+", x)
    if m is not None:
        sep = m.groups()[0]     
        return x.replace(sep, r"×")
    return x

def add_level(df, name, idx_name=None):
    names = [idx_name] + list(df.index.names)
    df = pd.concat({name:df})
    df.index.names = names
    return df
