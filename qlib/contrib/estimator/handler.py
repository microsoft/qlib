# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import abc
import bisect
import logging

import pandas as pd
import numpy as np

from ...log import get_module_logger, TimeInspector
from ...data import D
from ...utils import parse_config, transform_end_date

from . import processor as processor_module


class BaseDataHandler(abc.ABC):
    def __init__(self, processors=[], **kwargs):
        """
        :param start_date:
        :param end_date:
        :param kwargs:
        """
        # Set logger
        self.logger = get_module_logger("DataHandler")

        # init data using kwargs
        self._init_kwargs(**kwargs)

        # Setup data.
        self.raw_df, self.feature_names, self.label_names = self._init_raw_df()

        # Setup preprocessor
        self.processors = []
        for klass in processors:
            if isinstance(klass, str):
                try:
                    klass = getattr(processor_module, klass)
                except:
                    raise ValueError("unknown Processor %s" % klass)
            self.processors.append(klass(self.feature_names, self.label_names, **kwargs))

    def _init_kwargs(self, **kwargs):
        """
        init the kwargs of DataHandler
        """
        pass

    def _init_raw_df(self):
        """
        init raw_df, feature_names, label_names of DataHandler
        if the index of df_feature and df_label are not same, user need to overload this method to merge (e.g. inner, left, right merge).

        """
        df_features = self.setup_feature()
        feature_names = df_features.columns

        df_labels = self.setup_label()
        label_names = df_labels.columns

        raw_df = df_features.merge(df_labels, left_index=True, right_index=True, how="left")

        return raw_df, feature_names, label_names

    def reset_label(self, df_labels):
        for col in self.label_names:
            del self.raw_df[col]
        self.label_names = df_labels.columns
        self.raw_df = self.raw_df.merge(df_labels, left_index=True, right_index=True, how="left")

    def split_rolling_periods(
        self,
        train_start_date,
        train_end_date,
        validate_start_date,
        validate_end_date,
        test_start_date,
        test_end_date,
        rolling_period,
        calendar_freq="day",
    ):
        """
        Calculating the Rolling split periods, the period rolling on market calendar.
        :param train_start_date:
        :param train_end_date:
        :param validate_start_date:
        :param validate_end_date:
        :param test_start_date:
        :param test_end_date:
        :param rolling_period:  The market period of rolling
        :param calendar_freq: The frequence of the market calendar
        :yield: Rolling split periods
        """

        def get_start_index(calendar, start_date):
            start_index = bisect.bisect_left(calendar, start_date)
            return start_index

        def get_end_index(calendar, end_date):
            end_index = bisect.bisect_right(calendar, end_date)
            return end_index - 1

        calendar = self.raw_df.index.get_level_values("datetime").unique()

        train_start_index = get_start_index(calendar, pd.Timestamp(train_start_date))
        train_end_index = get_end_index(calendar, pd.Timestamp(train_end_date))
        valid_start_index = get_start_index(calendar, pd.Timestamp(validate_start_date))
        valid_end_index = get_end_index(calendar, pd.Timestamp(validate_end_date))
        test_start_index = get_start_index(calendar, pd.Timestamp(test_start_date))
        test_end_index = test_start_index + rolling_period - 1

        need_stop_split = False

        bound_test_end_index = get_end_index(calendar, pd.Timestamp(test_end_date))

        while not need_stop_split:

            if test_end_index > bound_test_end_index:
                test_end_index = bound_test_end_index
                need_stop_split = True

            yield (
                calendar[train_start_index],
                calendar[train_end_index],
                calendar[valid_start_index],
                calendar[valid_end_index],
                calendar[test_start_index],
                calendar[test_end_index],
            )

            train_start_index += rolling_period
            train_end_index += rolling_period
            valid_start_index += rolling_period
            valid_end_index += rolling_period
            test_start_index += rolling_period
            test_end_index += rolling_period

    def get_rolling_data(
        self,
        train_start_date,
        train_end_date,
        validate_start_date,
        validate_end_date,
        test_start_date,
        test_end_date,
        rolling_period,
        calendar_freq="day",
    ):
        # Set generator.
        for period in self.split_rolling_periods(
            train_start_date,
            train_end_date,
            validate_start_date,
            validate_end_date,
            test_start_date,
            test_end_date,
            rolling_period,
            calendar_freq,
        ):
            (
                x_train,
                y_train,
                x_validate,
                y_validate,
                x_test,
                y_test,
            ) = self.get_split_data(*period)
            yield x_train, y_train, x_validate, y_validate, x_test, y_test

    def get_split_data(
        self,
        train_start_date,
        train_end_date,
        validate_start_date,
        validate_end_date,
        test_start_date,
        test_end_date,
    ):
        """
        all return types are DataFrame
        """
        ## TODO: loc can be slow, expecially when we put it at the second level index.
        if self.raw_df.index.names[0] == "instrument":
            df_train = self.raw_df.loc(axis=0)[:, train_start_date:train_end_date]
            df_validate = self.raw_df.loc(axis=0)[:, validate_start_date:validate_end_date]
            df_test = self.raw_df.loc(axis=0)[:, test_start_date:test_end_date]
        else:
            df_train = self.raw_df.loc[train_start_date:train_end_date]
            df_validate = self.raw_df.loc[validate_start_date:validate_end_date]
            df_test = self.raw_df.loc[test_start_date:test_end_date]

        TimeInspector.set_time_mark()
        df_train, df_validate, df_test = self.setup_process_data(df_train, df_validate, df_test)
        TimeInspector.log_cost_time("Finished setup processed data.")

        x_train = df_train[self.feature_names]
        y_train = df_train[self.label_names]

        x_validate = df_validate[self.feature_names]
        y_validate = df_validate[self.label_names]

        x_test = df_test[self.feature_names]
        y_test = df_test[self.label_names]

        return x_train, y_train, x_validate, y_validate, x_test, y_test

    def setup_process_data(self, df_train, df_valid, df_test):
        """
        process the train, valid and test data
        :return: the processed train, valid and test data.
        """
        for processor in self.processors:
            df_train, df_valid, df_test = processor(df_train, df_valid, df_test)
        return df_train, df_valid, df_test

    def get_origin_test_label_with_date(self, test_start_date, test_end_date, freq="day"):
        """Get origin test label

        :param test_start_date: test start date
        :param test_end_date: test end date
        :param freq: freq
        :return: pd.DataFrame
        """
        test_end_date = transform_end_date(test_end_date, freq=freq)
        return self.raw_df.loc[(slice(None), slice(test_start_date, test_end_date)), self.label_names]

    @abc.abstractmethod
    def setup_feature(self):
        """
        Implement this method to load raw feature.
            the format of the feature is below
        return: df_features
        """
        pass

    @abc.abstractmethod
    def setup_label(self):
        """
        Implement this method to load and calculate label.
            the format of the label is below

        return: df_label
        """
        pass


class QLibDataHandler(BaseDataHandler):
    def __init__(self, start_date, end_date, *args, **kwargs):
        # Dates.
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(*args, **kwargs)

    def _init_kwargs(self, **kwargs):

        # Instruments
        instruments = kwargs.get("instruments", None)
        if instruments is None:
            market = kwargs.get("market", "csi500").lower()
            data_filter_list = kwargs.get("data_filter_list", list())
            self.instruments = D.instruments(market, filter_pipe=data_filter_list)
        else:
            self.instruments = instruments

        # Config of features and labels
        self._fields = kwargs.get("fields", [])
        self._names = kwargs.get("names", [])
        self._labels = kwargs.get("labels", [])
        self._label_names = kwargs.get("label_names", [])

        # Check arguments
        assert len(self._fields) > 0, "features list is empty"
        assert len(self._labels) > 0, "labels list is empty"

        # Check end_date
        # If test_end_date is -1 or greater than the last date, the last date is used
        self.end_date = transform_end_date(self.end_date)

    def setup_feature(self):
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

    def setup_label(self):
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

    def __init__(self, start_date, end_date, processors=None, **kwargs):
        if processors is None:
            processors = ["ConfigSectionProcessor"]  # default processor
        super().__init__(start_date, end_date, processors, **kwargs)

    def _init_kwargs(self, **kwargs):
        config = self.config_template.copy()
        if "config_update" in kwargs:
            config.update(kwargs["config_update"])
        fields, names = parse_config_to_fields(config)
        kwargs["fields"] = fields
        kwargs["names"] = names
        if "labels" not in kwargs:
            kwargs["labels"] = ["Ref($vwap, -2)/Ref($vwap, -1) - 1"]
        super()._init_kwargs(**kwargs)


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
        'kbar': {},
        'price': {
            'windows': [0],
            'feature': ['OPEN', 'HIGH', 'LOW', 'CLOSE'],
        },
        'rolling': {}
    }

    def _init_kwargs(self, **kwargs):
        kwargs['labels'] = ["Ref($close, -2)/Ref($close, -1) - 1"]
        super(Alpha158, self)._init_kwargs(**kwargs)


# if __name__ == '__main__':
#     import qlib
#
#     qlib.init()
#
#     handler = ALPHA80('2010-01-01', '2018-12-31')
#     data = handler.get_split_data(
#         pd.Timestamp('2010-01-01'), pd.Timestamp('2014-01-01'),
#         pd.Timestamp('2015-01-01'), pd.Timestamp('2016-01-01'),
#         pd.Timestamp('2017-01-01'), pd.Timestamp('2018-01-01'))
#     print(data[0])
#     data[0].to_pickle('alpha80.pkl')
