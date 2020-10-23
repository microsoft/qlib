# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor, MinMaxNorm, ZscoreNorm
from ...utils import get_cls_kwargs
from ...data.dataset import processor as processor_module
from ...log import TimeInspector
import copy


class ALPHA360(DataHandlerLP):
    def __init__(self, instruments="csi500", start_time=None, end_time=None):
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": {
                        "price": {
                            "windows": range(60)
                        },
                        "volume": {
                            "windows": range(60)
                        },
                    },
                    "label": self.get_label_config()
                },
                "group_fields": True,
            }
        }
        infer_processors = ["ConfigSectionProcessor"]  # ConfigSectionProcessor will normalize LABEL0
        super().__init__(instruments, start_time, end_time, data_loader=data_loader, infer_processors=infer_processors)

    def get_label_config(self):
        return (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])


class ALPHA360vwap(ALPHA360):
    def get_label_config(self):
        return (["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"])


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=["DropnaLabel", {
            "class": "CSZScoreNorm",
            "kwargs": {
                "fields_group": "label"
            }
        }],
        fit_start_time=None,
        fit_end_time=None,
    ):
        def check_transform_proc(proc_l):
            new_l = []
            for p in proc_l:
                if not isinstance(p, Processor):
                    klass, pkwargs = get_cls_kwargs(p, processor_module)
                    # FIXME: It's hard code here!!!!!
                    if isinstance(klass, (MinMaxNorm, ZscoreNorm)):
                        assert (fit_start_time is not None and fit_end_time is not None)
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

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": self.get_label_config()
                },
                "group_fields": True,
            }
        }
        super().__init__(instruments,
                         start_time,
                         end_time,
                         data_loader=data_loader,
                         infer_processors=infer_processors,
                         learn_processors=learn_processors)

    def get_feature_config(self):
        return {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }

    def get_label_config(self):
        return (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return (["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"])
