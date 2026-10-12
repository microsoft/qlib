# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.contrib.data.adanos_features import get_adanos_feature_config
from qlib.contrib.data.handler import Alpha158


class Alpha158AdanosUS(Alpha158):
    def __init__(self, instruments="sp500", **kwargs):
        super().__init__(instruments=instruments, **kwargs)

    def get_feature_config(self):
        base_fields, base_names = super().get_feature_config()
        sentiment_fields, sentiment_names = get_adanos_feature_config()
        return base_fields + sentiment_fields, base_names + sentiment_names
