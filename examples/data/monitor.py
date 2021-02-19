"""
This script is the demonstrating the implementation of following requirements.


NOTE: A lot of details is not considered in this script
- Corner case that will raise error( std == 0)


· Transformer:
	1) Basic statistics on different slices of the DataFrame df:
		§ The statistics include:
			· STD, Mean, Skewnes, Kurtosis
		§ The above statistics can be calculated on the following data slices:
			· df.groupby(['datetime'])
			· df.groupby(['datetime', 'industry' ])
			· df.groupby(['instrument', 'factor'])
			· df.apply("<expresion>").groupby([..]), in which [..] could be any one of the above slicing rules.
	2) Advanced statistics on different slices of the DataFrame df:
		§ Auto-correlation:
			· Calculate corr(df.loc[t, :, :], df.loc[t-w, :, :]), w=1, 2, ….
		§ Correlation between factors:
			· For any pair of factors (i, j): calculate corr(df.loc[t, :, i], df.loc[t, :,  j]). The result is a correlation matrix with each element corresponds to a correlation value between a pair of factors.
		§ The data slices are the same as those in 1).
· Monitor:
	1) Algorithms:
		§ Basic checks:  NaN.
		§ Point anomaly detection.
		§ Segment anomaly detection.
	2) Scenarios:
		§ Online anomaly detection: monitoring streaming data.
Offline anomaly detection: verifying whole historical data.


2021-2-19:

Effectiveness metrics
- Standard metrics:
    - [X] IC(Information Coefficient)  #case_3_1
    - [ ] IR(Information Ratio): Informatio Ratio is related to backest
    - [X] RankIC   #case_3_3
"""

# AUTO download data
from qlib.utils import exists_qlib_data
from qlib.tests.data import GetData
from qlib.config import REG_CN

provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
if not exists_qlib_data(provider_uri):
    print(f"Qlib data is not found in {provider_uri}")
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

import qlib

qlib.init()
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.monitor.metric import format_conv
from qlib.data.monitor.metric import MeanM, SkewM, KurtM, StdM, AutoCM, CorrM
from qlib.data.monitor.detector import NDDetector, SWNDD, ThresholdD
from qlib.data import D
import fire


UNIVERSE = "csi300"
START_TIME = "20200101"


def get_factor_df(col_idx=0):
    dh = Alpha158(instruments=UNIVERSE, infer_processors=[], learn_processors=[], start_time=START_TIME)
    df = dh.fetch()

    print(df.head())

    # We don't have industries in dataframe, we generate the with fake data
    industry = pd.Series(df.index.get_level_values("instrument").str.slice(stop=2).to_list(), index=df.index)

    # select a factor
    factor_df = format_conv(df.iloc[:, col_idx], industry=industry)
    print(f"Selected metric: {df.columns[col_idx]}")

    print(factor_df)
    return factor_df


def case_1_1():
    factor_df = get_factor_df()
    # 1) Extract metrics

    # 1.1) df.groupby(["datetime"])
    mtrc = MeanM()
    m_mean = mtrc.extract(factor_df)
    print(m_mean)

    ndd = NDDetector()
    ndd.fit(m_mean)  # use historical data to fit detector
    check_res = ndd.check(m_mean)
    print(check_res)  #  detecting on new data or historical data
    print(check_res.value_counts())


def case_1_2():
    factor_df = get_factor_df()
    # 1.2) df.groupby("datetime", "industry")
    mtrc = MeanM(group=["industry"])
    m_multi = mtrc.extract(factor_df)
    print(m_multi)

    for col_name, s in m_multi.iteritems():
        print(col_name)
        ndd = NDDetector()
        ndd.fit(s)  # use historical data to fit detector
        check_res = ndd.check(s)
        print(check_res)  #  detecting on new data or historical data
        print(check_res.value_counts())


def case_1_3_1_4():
    # case 1.3 and case 1.4
    # factor_df = get_factor_df()
    qdl = QlibDataLoader(config=(["$close/Ref($close, 1) - 1"], ["return"]))
    df = qdl.load(instruments=["SH600519"], start_time=START_TIME)
    df = format_conv(df)
    s = df.iloc[:, 0]
    print(s)
    dtc = SWNDD(window=20)
    dtc.fit(s)  # fit use historical data (TODO: updating will be supported in the future)
    check_res = dtc.check(s)  #
    print(check_res)
    print(check_res.value_counts())
    print(check_res[check_res])


def case_2_1():
    # · Calculate corr(df.loc[t, :, :], df.loc[t-w, :, :]), w=1, 2, ….
    factor_df = get_factor_df()
    acm = AutoCM()
    mtrc = acm.extract(factor_df)
    print(mtrc)

    thd = ThresholdD(0.0, reverse=True)
    check_res = thd.check(mtrc)

    print(check_res)
    print(check_res.value_counts())


def case_2_2():
    factor_df1, factor_df2 = get_factor_df(0), get_factor_df(1)

    cm = CorrM()
    mtrc = cm.extract(factor_df1, factor_df2)
    print(mtrc)

    thd = ThresholdD(0.0, reverse=True)
    check_res = thd.check(mtrc)

    print(check_res)
    print(check_res.value_counts())


def get_target(horizon=5):
    target = f"Ref($close, -{horizon + 1})/Ref($close, -1) - 1"  # There are lots of targets: return is one of them
    qdl = QlibDataLoader(config=([target], ["target"]))
    df = qdl.load(instruments=UNIVERSE, start_time=START_TIME)  # Aligning with factor will improve performance
    df = format_conv(df["target"])
    return df


def case_3_1_3_3():
    target, factor = get_target(), get_factor_df(0)
    ic_m, rank_ic_m = CorrM(), CorrM(mode="spearman")
    ic, rank_ic = ic_m.extract(factor, target), rank_ic_m.extract(factor, target)
    print(pd.DataFrame({"ic": ic, "rank_ic": rank_ic}))


def run(test_list=["case_1_1", "case_1_2", "case_1_3_1_4", "case_2_1", "case_2_2", "case_3_1_3_3"]):
    """
    run the specific tests

    python monitor.py case_3_1_3_3

    Parameters
    ----------
    test_list :  str[]
        The tests to run
    """
    if isinstance(test_list, str):
        test_list = [test_list]
    for fn in test_list:
        globals()[fn]()


if __name__ == "__main__":
    fire.Fire(run)
