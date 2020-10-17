# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import abc
import bisect
import logging
from typing import Union

import pandas as pd
import numpy as np

from ...log import get_module_logger, TimeInspector
from ...data import D
from ...config import C
from ...utils import parse_config, transform_end_date
from ...utils.serial import Serializable
from pathlib import Path

from . import processor as processor_module


# TODO: A more general handler interface which does not relies on internal pd.DataFrame is needed.
class DataHandler(Serializable):
    '''
    The steps to using a handler
    1. initialized data handler  (call by `init`).
    2. use the data


    The data handler try to maintain a handler with 2 level.
    `datetime` & `instruments`.
    
    Any order of the index level can be suported(The order will implied in the data).
    The order  <`datetime`, `instruments`> will be used when the dataframe index name is missed.

    Example of the data:

                              $close     $volume  Ref($close, 1)  Mean($close, 3)  $high-$low
    datetime   instrument
    2010-01-04 SH600000    81.807068  17145150.0       83.737389        83.016739    2.741058
               SH600004    13.313329  11800983.0       13.313329        13.317701    0.183632
               SH600005    37.796539  12231662.0       38.258602        37.919757    0.970325
               SH600006    22.672380   7095624.0       22.508326        22.573947    0.557785

    '''
    def __init__(self, init_data=True):
        # Set logger
        self.logger = get_module_logger("DataHandler")

        # Setup data.
        self._data = {}
        if init_data:
            self.init()
        super().__init__()

    def init(self, force_reload: bool=True):
        """
        initialize the data.
        In case of running intialization for multiple time, it will do nothing for the second time.

        Parameters
        ----------
        force_reload : bool
            force to reload the data even if the data have been initialized
        """
        pass
        # if force_reload or hasattr(self, '_initialized', False):

    def get_level_index(self, df: pd.DataFrame, level=Union[str, int]) -> int:
        """

        get the level index of `df` given `level`

        Parameters
        ----------
        df : pd.DataFrame
            data
        level : Union[str, int]
            index level

        Returns
        -------
        int:
            The level index in the multiple index
        """
        if isinstance(level, str):
            try:
                return df.index.names.index(level)
            except (AttributeError, ValueError):
                # NOTE: If level index is not given in the data, the default level index will be ('datetime', 'instrument') 
                return ('datetime', 'instrument').index(level)
        elif isinstance(level, int):
            return level
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def _fetch_df(self, df: pd.DataFrame, selector: Union[pd.Timestamp, slice, str, list], level: Union[str, int]):
        """
        fetch data from `data` with `selector` and `level`

        Parameters
        ----------
        df : pd.DataFrame
            the data frame to be selected
        selector : Union[pd.Timestamp, slice, str, list]
            selector
        level : Union[pd.Timestamp, slice, str]
            the level to use the selector
        """
        # Try to get the right index
        idx_slc = (selector, slice(None, None))
        if self.get_level_index(df, level) == 1:
                idx_slc = idx_slc[1], idx_slc[0]
        return df.loc(axis=0)[idx_slc]
        
    def fetch(self, selector: Union[pd.Timestamp, slice, str], level='datetime', key=None) -> Union[pd.DataFrame, dict]:
        if key is None:
            res = {}
            for k, df in self._data.items():
                res[k] = self._fetch_df(df, selector, level)
        else:
            res = self._fetch_df(self._data[key], selector, level)
        return res


class DataHandlerLP(DataHandler):
    '''
    DataHandler with **(L)earnable (P)rocessor**
    '''
    # data key
    DK_R = 'raw'
    DK_I = 'infer'
    DK_L = 'learn'

    # process type
    PTYPE_I = 'independent'
    # - _proc_infer_df will processed by infer_processors
    # - _proc_learn_df will be processed by learn_processors
    PTYPE_A = 'append'
    # - _proc_infer_df will processed by infer_processors
    # - _proc_learn_df will be processed by infer_processors + learn_processors
    #   - (e.g. _proc_infer_df processed by learn_processors )

    def __init__(self, infer_processors=[], learn_processors=[], process_type=PTYPE_A, **kwargs):
        """

        Parameters
        ----------
        infer_processors : list
            list of <description info> of processors to generate data for inference
            example of <description info>: 
            1) classname & kwargs:
                {
                    "class": "MinMaxNorm",
                    "kwargs": {
                        "fit_start_time": "20080101",
                        "fit_end_time": "20121231"
                    }
                }
            2) Only classname:
                "DropnaFeature"
            3) object instance of Processor

        learn_processors : list
            similar to infer_processors, but for generating data for learning models

        process_type: str
            PTYPE_I = 'independent'
            - _proc_infer_df will processed by infer_processors
            - _proc_learn_df will be processed by learn_processors
            PTYPE_A = 'append'
            - _proc_infer_df will processed by infer_processors
            - _proc_learn_df will be processed by infer_processors + learn_processors
              - (e.g. _proc_infer_df processed by learn_processors )
        """

        # Setup preprocessor
        self.infer_processors = []  # for lint
        self.learn_processors = []  # for lint
        for pname in 'infer_processors', 'learn_processors':
            for proc in locals()[pname]:
                getattr(self, pname).append(processor_module.init_proc_obj(proc))

        self.process_type = process_type
        super().__init__(**kwargs)

    def get_all_processors(self):
        return self.infer_processors + self.learn_processors

    def _init_raw_data(self):
        """
        initialize the raw data
        the raw data will be saved in to `self._data['raw']`
        """
        raise NotImplementedError(f"Please implement the `_init_raw_data` method")

    def fit(self):
        for proc in self.get_all_processors():
            proc.fit(self)

    def fit_process_data(self):
        """
        fit and process data

        The input of the `fit` will be the output of the previous processor
        """
        self.process_data(with_fit=True)
        

    def process_data(self, with_fit: bool=False):
        """
        process_data data. Fun `processor.fit` if necessary

        Parameters
        ----------
        with_fit : bool
            The input of the `fit` will be the output of the previous processor
        """
        # data for inference
        _infer_df = self._data[DataHandlerLP.DK_R]
        for proc in self.infer_processors:
            if not proc.is_for_infer():
                raise TypeError("Only processors usable for inference can be used in `infer_processors` ")
            if with_fit:
                proc.fit(self, _infer_df)
            _infer_df = proc(_infer_df)

        # data for learning
        if self.process_type == DataHandlerLP.PTYPE_I:
            _learn_df = self._data[DataHandlerLP.DK_R]
        elif self.process_type == DataHandlerLP.PTYPE_A:
            # based on `infer_df` and append the processor
            _learn_df = _infer_df
        else:
            raise NotImplementedError(f"This type of input is not supported")

        for proc in self.learn_processors:
            if with_fit:
                proc.fit(self, _learn_df)
            _learn_df = proc(_learn_df)
        
        self._data.update({
            DataHandlerLP.DK_I: _infer_df,
            DataHandlerLP.DK_L: _learn_df,
        })

    # init type
    IT_FIT_SEQ = 'fit_seq'  # the input of `fit` will be the output of the previous processor
    IT_FIT_IND = 'fit_ind'  # the input of `fit` will be the original df
    IT_LS = 'load_state'  # The state of the object has been load by pickle
    
    def init(self, init_type: str=IT_FIT_SEQ, path: Path=None):
        """
        Initialize the data of Qlib

        Parameters
        ----------
        init_type : str
            'fit' or 'load_state'
        path : path
            if `init_type` == 'load_state': `path` will be used to load_state
        """
        self._init_raw_data()

        if init_type == DataHandlerLP.IT_FIT_IND:
            self.fit()
            self.process_data()
        elif init_type == DataHandlerLP.IT_LS:
            self.process_data()
        elif init_type == DataHandlerLP.IT_FIT_SEQ:
            self.fit_process_data()
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # TODO: Be able to cache handler data. Save the memory for data processing


class DataHandlerLPWL(DataHandlerLP):
    '''
    DataHandler with (L)earnable (P)rocessor with (L)abel
    '''

    def _init_raw_data(self):
        """
        init raw_df, feature_names, label_names of DataHandler
        if the index of df_feature and df_label are not same, user need to overload this method to merge (e.g. inner, left, right merge).
        """
        df_features = self.load_feature()
        feature_names = df_features.columns

        df_labels = self.load_label()
        label_names = df_labels.columns

        raw_df = df_features.merge(df_labels, left_index=True, right_index=True, how="left")
        self.feature_names = feature_names
        self.label_names = label_names
        self._data['raw'] = raw_df

    def load_feature(self):
        """
        Implement this method to load raw feature.
            the format of the feature is below
        return: df_features
        """
        raise NotImplementedError(f"Please implement `load_feature`")

    def load_label(self):
        """
        Implement this method to load and calculate label.
            the format of the label is below

        return: df_label
        """
        raise NotImplementedError(f"Please implement `load_label`")

    def get_feature_names(self):
        return self.feature_names

    def get_label_names(self):
        return self.label_names


class QLibDataHandler(DataHandlerLPWL):
    def __init__(self, start_date, end_date, *args, **kwargs):
        # Dates.
        self.start_date = start_date
        self.end_date = end_date

        # Instruments
        instruments = kwargs.pop("instruments", None)
        if instruments is None:
            market = kwargs.pop("market", "csi500").lower()
            data_filter_list = kwargs.pop("data_filter_list", list())
            self.instruments = D.instruments(market, filter_pipe=data_filter_list)
        else:
            self.instruments = instruments

        # Config of features and labels
        self._fields = kwargs.pop("fields", [])
        self._names = kwargs.pop("names", [])
        self._labels = kwargs.pop("labels", [])
        self._label_names = kwargs.pop("label_names", [])

        # Check arguments
        assert len(self._fields) > 0, "features list is empty"
        assert len(self._labels) > 0, "labels list is empty"

        # Check end_date
        # If test_end_date is -1 or greater than the last date, the last date is used
        self.end_date = transform_end_date(self.end_date)

        super().__init__(*args, **kwargs)

    def load_feature(self):
        """
        Load the raw data.
        return: df_features
        """
        TimeInspector.set_time_mark()

        if len(self._names) == 0:
            names = ["F%d" % i for i in range(len(self._fields))]
        else:
            names = self._names

        df_features = D.features(self.instruments, self._fields, self.start_date, self.end_date)
        df_features.columns = names

        TimeInspector.log_cost_time("Finished loading features.")

        return df_features

    def load_label(self):
        """
        Build up labels in df through users' method
        :return:  df_labels
        """
        TimeInspector.set_time_mark()

        if len(self._label_names) == 0:
            label_names = ["LABEL%d" % i for i in range(len(self._labels))]
        else:
            label_names = self._label_names

        df_labels = D.features(self.instruments, self._labels, self.start_date, self.end_date)
        df_labels.columns = label_names

        TimeInspector.log_cost_time("Finished loading labels.")

        return df_labels


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
                "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                for d in windows
            ]
            names += ["VSUMD%d" % d for d in windows]

    return fields, names


class ConfigQLibDataHandler(QLibDataHandler):
    config_template = {}  # template

    def __init__(self, start_date, end_date, infer_processors=["ConfigSectionProcessor"], learn_processors=[], **kwargs):
        config = self.config_template.copy()
        if "config_update" in kwargs:
            config.update(kwargs["config_update"])
        fields, names = parse_config_to_fields(config)
        kwargs["fields"] = fields
        kwargs["names"] = names
        if "labels" not in kwargs:
            kwargs["labels"] = ["Ref($vwap, -2)/Ref($vwap, -1) - 1"]

        super().__init__(start_date, end_date, infer_processors=infer_processors, learn_processors=learn_processors, **kwargs)
