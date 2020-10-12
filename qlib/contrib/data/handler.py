# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...data.dataset.handler import ConfigQLibDataHandler
from ...log import TimeInspector


class ALPHA360(ConfigQLibDataHandler):
    config_template = {
        "price": {"windows": range(60)},
        "volume": {"windows": range(60)},
    }


class QLibDataHandlerV1(ConfigQLibDataHandler):
    config_template = {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
        },
        "rolling": {},
    }

    def __init__(self, start_date, end_date, processors=None, **kwargs):
        if processors is None:
            processors = ["PanelProcessor"]  # V1 default processor
        super().__init__(start_date, end_date, processors, **kwargs)

    def setup_label(self):
        """
        load the labels df
        :return:  df_labels
        """
        TimeInspector.set_time_mark()

        df_labels = super().setup_label()

        ## calculate new labels
        df_labels["LABEL1"] = df_labels["LABEL0"].groupby(level="datetime").apply(lambda x: (x - x.mean()) / x.std())

        df_labels = df_labels.drop(["LABEL0"], axis=1)

        TimeInspector.log_cost_time("Finished loading labels.")

        return df_labels


class Alpha158(QLibDataHandlerV1):
    config_template = {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW", "CLOSE"],
        },
        "rolling": {},
    }

    def _init_kwargs(self, **kwargs):
        kwargs["labels"] = ["Ref($close, -2)/Ref($close, -1) - 1"]
        super(Alpha158, self)._init_kwargs(**kwargs)


