#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Interfaces to interpret models
"""

import pandas as pd
from abc import abstractmethod


class FeatureInt:
    """Feature (Int)erpreter"""

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """get feature importance

        Returns
        -------
            The index is the feature name.

            The greater the value, the higher importance.
        """


class LightGBMFInt(FeatureInt):
    """LightGBM (F)eature (Int)erpreter"""

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance

        Notes
        -----
            parameters reference:
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance
        """
        return pd.Series(self.model.feature_importance(*args, **kwargs), index=self.model.feature_name()).sort_values(
            ascending=False
        )
