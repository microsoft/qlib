# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
import pandas as pd
from qlib.data import D
from typing import Tuple


class DataLoader(ABC):
    '''
    DataLoader is designed for loading raw data from original data source.
    '''
    @abstractmethod
    def load(self, instruments, start_time=None, end_time=None) -> pd.DataFrame:
        """
        load the data as pd.DataFrame

        Returns
        -------
        pd.DataFrame:
            data load from the under layer source

            Example of the data:
    The multi-index of the columns is optional.
                             feature                                                            label
                              $close     $volume  Ref($close, 1)  Mean($close, 3)  $high-$low  LABEL0
    datetime   instrument
    2010-01-04 SH600000    81.807068  17145150.0       83.737389        83.016739    2.741058  0.0032
               SH600004    13.313329  11800983.0       13.313329        13.317701    0.183632  0.0042
               SH600005    37.796539  12231662.0       38.258602        37.919757    0.970325  0.0289
        """
        pass


class QlibDataLoader(DataLoader):
    '''Same as QlibDataLoader. The fields can be define by config'''
    def __init__(self, config: Tuple[list, tuple, dict], group_fields: bool = False, filter_pipe=None):
        """
        Parameters
        ----------
        config : Tuple[list ,tuple, dict]
            Config will be used to describe the fields and column names

            if `group_fields`:
                <config> := {
                    "group_name1": <fields_info1>
                    "group_name2": <fields_info2>
                }
            else:
                 <config> := <fields_info>

            <fields_info> := ["expr", ...] | (["expr", ...], ["col_name", ...]) | <fields_info_config>

            <fields_info_config> is a config with dict type which could be parsed by `parse_config_to_fields`

            Here is a few examples to describe the fields
            TODO:

        group_fields : bool
            Will the fields be grouped. Multi-index will be used for the group
        """
        if group_fields:
            fields_all = []
            name_grp_info = []
            for grp, fields_info in config.items():
                fields, names = self._parse_fields_info(fields_info)
                fields_all.extend(fields)
                name_grp_info.extend([(grp, n) for n in names])
            self.fields, self.names = fields_all, name_grp_info
        else:
            self.fields, self.names = self._parse_fields_info(fields_info)

        self.group_fields = group_fields
        self.filter_pipe = filter_pipe

    def _parse_fields_info(self, fields_info: Tuple[list, tuple, dict]) -> Tuple[list, list]:
        if isinstance(fields_info, dict):
            fields, names = parse_config_to_fields(fields_info)
        elif isinstance(fields_info, list):
            fields = fields_info
            names = fields
        elif isinstance(fields_info, tuple):
            fields, names = fields_info
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return fields, names

    def load(self,
             instruments,
             config: Tuple[list, tuple, dict],
             group_fields=False,
             start_time=None,
             end_time=None) -> Tuple[pd.DataFrame, dict]:
        df = D.features(D.instruments(instruments, filter_pipe=self.filter_pipe), self.fields, start_time, end_time)
        df.columns = pd.MultiIndex.from_tuples(self.names) if self.group_fields else self.names
        df = df.swaplevel().sort_index()
        return df


# TODO: make it easier to understand the config language
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
                "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d) for d in windows
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
                "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)" %
                (d, d) for d in windows
            ]
            names += ["WVMA%d" % d for d in windows]
        if use("VSUMP"):
            fields += [
                "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d)
                for d in windows
            ]
            names += ["VSUMP%d" % d for d in windows]
        if use("VSUMN"):
            fields += [
                "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d)
                for d in windows
            ]
            names += ["VSUMN%d" % d for d in windows]
        if use("VSUMD"):
            fields += [
                "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d) for d in windows
            ]
            names += ["VSUMD%d" % d for d in windows]

    return fields, names
