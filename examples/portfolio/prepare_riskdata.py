# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
import pandas as pd

from qlib.data import D
from qlib.model.riskmodel import StructuredCovEstimator


def prepare_data(riskdata_root="./riskdata", T=240, start_time="2016-01-01"):
    universe = D.features(D.instruments("csi300"), ["$close"], start_time=start_time).swaplevel().sort_index()

    price_all = (
        D.features(D.instruments("all"), ["$close"], start_time=start_time).squeeze().unstack(level="instrument")
    )

    # StructuredCovEstimator is a statistical risk model
    riskmodel = StructuredCovEstimator()

    for i in range(T - 1, len(price_all)):
        date = price_all.index[i]
        ref_date = price_all.index[i - T + 1]

        print(date)

        codes = universe.loc[date].index
        price = price_all.loc[ref_date:date, codes]

        # calculate return and remove extreme return
        ret = price.pct_change()
        ret.clip(ret.quantile(0.025), ret.quantile(0.975), axis=1, inplace=True)

        # run risk model
        F, cov_b, var_u = riskmodel.predict(ret, is_price=False, return_decomposed_components=True)

        # save risk data
        root = riskdata_root + "/" + date.strftime("%Y%m%d")
        os.makedirs(root, exist_ok=True)

        pd.DataFrame(F, index=codes).to_pickle(root + "/factor_exp.pkl")
        pd.DataFrame(cov_b).to_pickle(root + "/factor_cov.pkl")
        # for specific_risk we follow the convention to save volatility
        pd.Series(np.sqrt(var_u), index=codes).to_pickle(root + "/specific_risk.pkl")


if __name__ == "__main__":
    import qlib

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    prepare_data()
