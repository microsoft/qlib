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
                        "price": {"windows": range(60)},
                        "volume": {"windows": range(60)},
                    },
                    "label": self.get_label_config(),
                },
            },
        }
        infer_processors = [
            {"class": "ConfigSectionProcessor", "module_path": "qlib.contrib.data.processor"}
        ]  # ConfigSectionProcessor will normalize LABEL0
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
        learn_processors=["DropnaLabel", {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}],
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
                        assert fit_start_time is not None and fit_end_time is not None
                        pkwargs.update(
                            {
                                "fit_start_time": fit_start_time,
                                "fit_end_time": fit_end_time,
                            }
                        )
                    new_l.append({"class": klass.__name__, "kwargs": pkwargs})
                else:
                    new_l.append(p)
            return new_l

        infer_processors = check_transform_proc(infer_processors)
        learn_processors = check_transform_proc(learn_processors)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {"feature": self.get_feature_config(), "label": self.get_label_config()},
            },
        }
        super().__init__(
            instruments,
            start_time,
            end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return self.parse_config_to_fields(conf)

    def get_label_config(self):
        return (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])

    @staticmethod
    def parse_config_to_fields(config):
        """create factors from config

        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            }
        }
        """
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += ["Ref($volume, %d)/$volume" % d if d != 0 else "$volume/$volume" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field
            use = lambda x: x not in exclude and (include is None or x in include)
            if use("ROC"):
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("STD"):
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("QTLD"):
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RSV"):
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("CNTD"):
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMN"):
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]

        return fields, names


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return (["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"])
