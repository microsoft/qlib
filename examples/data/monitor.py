"""
This script is the demonstrating the implementation of Metric Extractor and Detector

NOTE: A lot of details is not considered in this script
- Corner case that will raise error( std == 0)



The following functions are used to demonstrate the following examples


· Metric Extractor:
	case 1) Basic statistics on different slices of the DataFrame df:
		1) The statistics include:
			· STD, Mean, Skewnes, Kurtosis
		2) The above statistics can be calculated on the following data slices:
			· df.groupby(['datetime'])
			· df.groupby(['datetime', 'industry' ])
                3) The statistics could be calculated on the time dimension for each instruments and factor(the factor can be represented by experssion)
			· <df implemented by expresion>.groupby(['instrument', 'factor'])
	case 2) Advanced statistics on different slices of the DataFrame df:
		1) Auto-correlation:
			· Calculate corr(df.loc[t, :, :], df.loc[t-w, :, :]), w=1, 2, ….
		2) Correlation between factors:
			· For any pair of factors (i, j): calculate corr(df.loc[t, :, i], df.loc[t, :,  j]). The result is a correlation matrix with each element corresponds to a correlation value between a pair of factors.

· Detector:  detect the abnormality of the extracted metric;
	a) Algorithms:
		§ Basic checks:  NaN.
		§ Point anomaly detection.
		§ Segment anomaly detection.
	b) Scenarios:
		§ Online anomaly detection: monitoring streaming data.
The usage of the detectors are demonstrated in the `case_1_*`and `case_2_*`


case 3): Examples to use MetricExt to monitor IC and rank IC
        1) IC(Information Coefficient)  #case_3_1
        2) RankIC   #case_3_2
"""

# AUTO download data
from typing import List, Union
from qlib.utils import exists_qlib_data
from qlib.tests.data import GetData
from qlib.config import REG_CN

provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
if not exists_qlib_data(provider_uri):
    print(f"Qlib data is not found in {provider_uri}")
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

import qlib
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

# ------------------ a helper function to get data to demonstrate the functionality --------------------


def get_data_df(col_idx: Union[int, List[int]] = 0, verbose: bool = True):
    """
    a helper function to get data to demonstrate the functionality.

    Parameters
    ----------
    col_idx : Union[int, List[int]]
        column index of the metrics
    """
    dh = Alpha158(instruments=UNIVERSE, infer_processors=[], learn_processors=[], start_time=START_TIME)
    df = dh.fetch()

    if verbose:
        print(df.head())

    # We don't have industries in dataframe, we generate the with fake data
    industry = pd.Series(df.index.get_level_values("instrument").str.slice(stop=2).to_list(), index=df.index)

    # select a factor
    factor_df = format_conv(df.iloc[:, col_idx], industry=industry)
    if verbose:
        print(f"Selected metric: {df.columns[col_idx]}")
        print(factor_df)
    return factor_df


def get_target(horizon=5):
    target = f"Ref($close, -{horizon + 1})/Ref($close, -1) - 1"  # There are lots of targets: return is one of them
    qdl = QlibDataLoader(config=([target], ["target"]))
    df = qdl.load(instruments=UNIVERSE, start_time=START_TIME)  # Aligning with factor will improve performance
    df = format_conv(df["target"])
    return df


# -----------------  Cases to demonstrate the usage of detector and examples ----------------------


def case_1_1():
    factor_df = get_data_df()
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
    factor_df = get_data_df()
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


def case_1_3():
    # case 1.3
    # factor_df = get_data_df()
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
    factor_df = get_data_df()
    acm = AutoCM()
    mtrc = acm.extract(factor_df)
    print(mtrc)

    thd = ThresholdD(0.0, reverse=True)
    check_res = thd.check(mtrc)

    print(check_res)
    print(check_res.value_counts())


def case_2_2():
    factor_df1, factor_df2 = get_data_df(0), get_data_df(1)

    cm = CorrM()
    mtrc = cm.extract(factor_df1, factor_df2)
    print(mtrc)

    thd = ThresholdD(0.0, reverse=True)
    check_res = thd.check(mtrc)

    print(check_res)
    print(check_res.value_counts())


def case_3_1_3_2():
    target, factor = get_target(), get_data_df(0)
    ic_m, rank_ic_m = CorrM(), CorrM(mode="spearman")
    ic, rank_ic = ic_m.extract(factor, target), rank_ic_m.extract(factor, target)
    print(pd.DataFrame({"ic": ic, "rank_ic": rank_ic}))


def run(test_list=["case_1_1", "case_1_2", "case_1_3", "case_2_1", "case_2_2", "case_3_1_3_2"]):
    """
    run the specific tests

    python monitor.py case_3_1_3_2

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
    qlib.init()
    fire.Fire(run)
