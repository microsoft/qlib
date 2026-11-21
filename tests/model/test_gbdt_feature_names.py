# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH


@pytest.mark.parametrize(
    ("columns", "expected_names"),
    [
        (["ROC5", "ROC10"], ["ROC5", "ROC10"]),
        (["Ref($close, 1)", "ROC10"], ["Column_0", "Column_1"]),
        (["factor one", "factor_one"], ["Column_0", "Column_1"]),
        (["ROC5", "ROC5"], ["Column_0", "Column_1"]),
        ([0, 1], ["Column_0", "Column_1"]),
    ],
)
def test_lgb_feature_importance_names(columns, expected_names):
    features = pd.DataFrame(np.arange(24).reshape(12, 2), columns=columns)
    frame = pd.concat({"feature": features, "label": features.iloc[:, :1]}, axis=1)
    dataset = Mock(spec=DatasetH, segments={"train": slice(None)})
    dataset.prepare.return_value = frame

    model = LGBModel(num_boost_round=2, num_threads=1, min_data_in_leaf=1)
    with patch("qlib.contrib.model.gbdt.R"):
        model.fit(dataset)

    assert model.model.feature_name() == expected_names
    assert set(model.get_feature_importance().index) == set(expected_names)
