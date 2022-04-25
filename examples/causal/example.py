# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Here is an example to leverage dowhy for causal estimate

1) install dowhy by following command
pip install -r requirements.txt

2) run this script

"""
import io
from dowhy import CausalModel
import pandas as pd
import numpy as np
import qlib
from qlib.utils import init_instance_by_config
from tqdm.auto import tqdm
import yaml

qlib.init()

np.random.seed(42)  # for get consistant result

# 1) loading data
conf = """
class: Alpha158
module_path: qlib.contrib.data.handler
kwargs:
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: csi300
    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSRankNorm
          kwargs:
              fields_group: label
"""
hconf = yaml.safe_load(io.StringIO(conf))
hd = init_instance_by_config(hconf)

# NOTE:
# The calculation of causation is very slow.
# So only a small slice of data is used
df = hd.fetch(slice("20100101", "20100131"), data_key=hd.DK_L)

# Outcome
label = "LABEL0"

df = df.iloc[:, :-1].sample(10, axis=1).assign(**{label: df[label]})

# 2) Causal Estimate for each feature

cause_dict = {}
for col in tqdm(df.columns):
    if col == label:  # skip label
        continue
    common_causes = df.columns[~df.columns.isin([col, label])].to_list()
    model = CausalModel(data=df, treatment=col, outcome=label, common_causes=common_causes)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    # Uncomment to get more detailed information
    # print(identified_estimand)
    # print(estimate)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print("Causal Estimate is " + str(estimate.value))
    cause_dict[col] = estimate.value

print(pd.Series(cause_dict).sort_values())
# At last, this script will output cause estimate like this.
"""
CORD20    -0.078231
MAX20     -0.052873
CNTP5     -0.036751
BETA10    -0.013858
CORD30    -0.008729
VSTD10    -0.007410
WVMA20    -0.002792
VSUMD30    0.013985
VMA5       0.019856
CORR20     0.025246
dtype: float64
"""

print(df.corr()[label].drop(label).sort_values())
# comparing with correlation
"""
CORD20    -0.045939
CORD30    -0.042149
CNTP5     -0.018027
WVMA20    -0.015618
BETA10    -0.014057
CORR20    -0.006562
VSUMD30   -0.004389
VSTD10    -0.002526
MAX20     -0.000700
VMA5       0.006577
Name: LABEL0, dtype: float64
"""
