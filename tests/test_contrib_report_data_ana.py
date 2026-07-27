import numpy as np
import pandas as pd

from qlib.contrib.report.data.ana import FeaInfAna


def test_fea_inf_ana_counts_inf_values_with_numpy_2():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "SH000001"),
            (pd.Timestamp("2024-01-01"), "SH000002"),
            (pd.Timestamp("2024-01-02"), "SH000001"),
            (pd.Timestamp("2024-01-02"), "SH000002"),
        ],
        names=["datetime", "instrument"],
    )
    dataset = pd.DataFrame(
        {
            "feature": [1.0, np.inf, -np.inf, 2.0],
            "clean": [1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )

    analyser = FeaInfAna(dataset)

    assert analyser._inf_cnt.loc[pd.Timestamp("2024-01-01"), "feature"] == 1
    assert analyser._inf_cnt.loc[pd.Timestamp("2024-01-02"), "feature"] == 1
    assert analyser.skip("clean")
