# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...data.dataset.handler import ConfigQLibDataHandler
from ...data.dataset.processor import Processor, MinMaxNorm, ZscoreNorm, get_cls_kwargs
from ...log import TimeInspector
import copy


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

    def __init__(self, start_date, end_date, infer_processors=[], learn_processors=["DropnaLabel"], fit_start_time=None, fit_end_time=None, **kwargs):
        def check_transform_proc(proc_l):
            new_l = []
            for p in proc_l:
                if not isinstance(p, Processor):
                    klass, pkwargs = get_cls_kwargs(p)
                    if isinstance(klass, (MinMaxNorm, ZscoreNorm)):
                        assert(fit_start_time is not None and fit_end_time is not None)
                        pkwargs.update({
                            "fit_start_time": fit_start_time,
                            "fit_end_time": fit_end_time,
                            })
                    new_l.append({"class": klass.__name__, "kwargs": pkwargs})
                else:
                    new_l.append(p)
            return new_l

        infer_processors = check_transform_proc(infer_processors)
        learn_processors = check_transform_proc(learn_processors)

        super().__init__(start_date, end_date, infer_processors=infer_processors, learn_processors=learn_processors, **kwargs)

    def load_label(self):
        """
        load the labels df
        :return:  df_labels
        """
        TimeInspector.set_time_mark()

        df_labels = super().load_label()

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

    def __init__(self, *args, **kwargs):
        kwargs["labels"] = ["Ref($close, -2)/Ref($close, -1) - 1"]
        super().__init__(*args, **kwargs)
