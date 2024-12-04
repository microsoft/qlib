import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
from tqdm.auto import tqdm

# tqdm.pandas()  # for progress_apply
# %matplotlib inline
# %load_ext autoreload


# # Meta Input

# +
with open("./internal_data_s20.pkl", "rb") as f:
    data = pickle.load(f)

data.data_ic_df.columns.names = ["start_date", "end_date"]

data_sim = data.data_ic_df.droplevel(axis=1, level="end_date")

data_sim.index.name = "test datetime"
# -

plt.figure(figsize=(40, 20))
sns.heatmap(data_sim)

plt.figure(figsize=(40, 20))
sns.heatmap(data_sim.rolling(20).mean())

# # Meta Model

from qlib import auto_init

auto_init()
from qlib.workflow import R

exp = R.get_exp(experiment_name="DDG-DA")
meta_rec = exp.list_recorders(rtype="list", max_results=1)[0]
meta_m = meta_rec.load_object("model")

pd.DataFrame(meta_m.tn.twm.linear.weight.detach().numpy()).T[0].plot()

pd.DataFrame(meta_m.tn.twm.linear.weight.detach().numpy()).T[0].rolling(5).mean().plot()

# # Meta Output

# +
with open("./tasks_s20.pkl", "rb") as f:
    tasks = pickle.load(f)

task_df = {}
for t in tasks:
    test_seg = t["dataset"]["kwargs"]["segments"]["test"]
    if None not in test_seg:
        # The last rolling is skipped.
        task_df[test_seg] = t["reweighter"].time_weight
task_df = pd.concat(task_df)

task_df.index.names = ["OS_start", "OS_end", "IS_start", "IS_end"]
task_df = task_df.droplevel(["OS_end", "IS_end"])
task_df = task_df.unstack("OS_start")
# -

plt.figure(figsize=(40, 20))
sns.heatmap(task_df.T)

plt.figure(figsize=(40, 20))
sns.heatmap(task_df.rolling(10).mean().T)

# # Sub Models
#
# NOTE:
# - this section assumes that the model is Linear model!!
# - Other models does not support this analysis

exp = R.get_exp(experiment_name="rolling_ds")


def show_linear_weight(exp):
    coef_df = {}
    for r in exp.list_recorders("list"):
        t = r.load_object("task")
        if None in t["dataset"]["kwargs"]["segments"]["test"]:
            continue
        m = r.load_object("params.pkl")
        coef_df[t["dataset"]["kwargs"]["segments"]["test"]] = pd.Series(m.coef_)

    coef_df = pd.concat(coef_df)

    coef_df.index.names = ["test_start", "test_end", "coef_idx"]

    coef_df = coef_df.droplevel("test_end").unstack("coef_idx").T

    plt.figure(figsize=(40, 20))
    sns.heatmap(coef_df)
    plt.show()


show_linear_weight(R.get_exp(experiment_name="rolling_ds"))

show_linear_weight(R.get_exp(experiment_name="rolling_models"))
