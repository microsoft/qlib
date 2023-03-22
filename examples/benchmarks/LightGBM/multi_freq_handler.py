#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pandas as pd

from qlib.data.dataset.loader import QlibDataLoader
from qlib.contrib.data.handler import DataHandlerLP, _DEFAULT_LEARN_PROCESSORS, check_transform_proc


class Avg15minLoader(QlibDataLoader):
    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        df = super(Avg15minLoader, self).load(instruments, start_time, end_time)
        if self.is_group:
            # feature_day(day freq) and feature_15min(1min freq, Average every 15 minutes) renamed feature
            df.columns = df.columns.map(lambda x: ("feature", x[1]) if x[0].startswith("feature") else x)
        return df


class Avg15minHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        data_loader = Avg15minLoader(
            config=self.loader_config(), filter_pipe=filter_pipe, freq=freq, inst_processors=inst_processors
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def loader_config(self):

        # Results for dataset: df: pd.DataFrame
        #   len(df.columns) == 6 + 6 * 16, len(df.index.get_level_values(level="datetime").unique()) == T
        #   df.columns: close0, close1, ..., close16, open0, ..., open16, ..., vwap16
        #       freq == day:
        #           close0, open0, low0, high0, volume0, vwap0
        #       freq == 1min:
        #           close1, ..., close16, ..., vwap1, ..., vwap16
        #   df.index.name == ["datetime", "instrument"]: pd.MultiIndex
        # Example:
        #                          feature                        ...                  label
        #                           close0      open0       low0  ... vwap1 vwap16    LABEL0
        # datetime   instrument                                   ...
        # 2020-10-09 SH600000    11.794546  11.819587  11.769505  ...   NaN    NaN -0.005214
        # 2020-10-15 SH600000    12.044961  11.944795  11.932274  ...   NaN    NaN -0.007202
        # ...                          ...        ...        ...  ...   ...    ...       ...
        # 2021-05-28 SZ300676     6.369684   6.495406   6.306568  ...   NaN    NaN -0.001321
        # 2021-05-31 SZ300676     6.601626   6.465643   6.465130  ...   NaN    NaN -0.023428

        # features day: len(columns) == 6, freq = day
        # $close is the closing price of the current trading day:
        #   if the user needs to get the `close` before the last T days, use Ref($close, T-1), for example:
        #                                    $close  Ref($close, 1)  Ref($close, 2)  Ref($close, 3)  Ref($close, 4)
        #         instrument datetime
        #         SH600519   2021-06-01  244.271530
        #                    2021-06-02  242.205917      244.271530
        #                    2021-06-03  242.229889      242.205917      244.271530
        #                    2021-06-04  245.421524      242.229889      242.205917      244.271530
        #                    2021-06-07  247.547089      245.421524      242.229889      242.205917      244.271530

        # WARNING: Ref($close, N), if N == 0, Ref($close, N) ==> $close

        fields = ["$close", "$open", "$low", "$high", "$volume", "$vwap"]
        # names: close0, open0, ..., vwap0
        names = list(map(lambda x: x.strip("$") + "0", fields))

        config = {"feature_day": (fields, names)}

        # features 15min: len(columns) == 6 * 16, freq = 1min
        #   $close is the closing price of the current trading day:
        #       if the user gets 'close' for the i-th 15min of the last T days, use `Ref(Mean($close, 15), (T-1) * 240 + i * 15)`, for example:
        #                                    Ref(Mean($close, 15), 225)  Ref(Mean($close, 15), 465)  Ref(Mean($close, 15), 705)
        #             instrument datetime
        #             SH600519   2021-05-31                  241.769897                  243.077942                  244.712997
        #                        2021-06-01                  244.271530                  241.769897                  243.077942
        #                        2021-06-02                  242.205917                  244.271530                  241.769897

        # WARNING: Ref(Mean($close, 15), N), if N == 0, Ref(Mean($close, 15), N) ==> Mean($close, 15)

        # Results of the current script:
        #   time:   09:00 --> 09:14,            ..., 14:45 --> 14:59
        #   fields: Ref(Mean($close, 15), 225), ..., Mean($close, 15)
        #   name:   close1,                     ..., close16
        #

        # Expression description: take close as an example
        #   Mean($close, 15) ==> df["$close"].rolling(15, min_periods=1).mean()
        #   Ref(Mean($close, 15), 15) ==> df["$close"].rolling(15, min_periods=1).mean().shift(15)

        #   NOTE: The last data of each trading day, which is the average of the i-th 15 minutes

        # Average:
        #   Average of the i-th 15-minute period of each trading day: 1 <= i <= 250 // 16
        #       Avg(15minutes): Ref(Mean($close, 15), 240 - i * 15)
        #
        #   Average of the first 15 minutes of each trading day; i = 1
        #       Avg(09:00 --> 09:14), df.index.loc["09:14"]: Ref(Mean($close, 15), 240- 1 * 15) ==> Ref(Mean($close, 15), 225)
        #   Average of the last 15 minutes of each trading day; i = 16
        #       Avg(14:45 --> 14:59), df.index.loc["14:59"]: Ref(Mean($close, 15), 240 - 16 * 15) ==> Ref(Mean($close, 15), 0) ==> Mean($close, 15)

        # 15min resample to day
        #   df.resample("1d").last()
        tmp_fields = []
        tmp_names = []
        for i, _f in enumerate(fields):
            _fields = [f"Ref(Mean({_f}, 15), {j * 15})" for j in range(1, 240 // 15)]
            _names = [f"{names[i][:-1]}{int(names[i][-1])+j}" for j in range(240 // 15 - 1, 0, -1)]
            _fields.append(f"Mean({_f}, 15)")
            _names.append(f"{names[i][:-1]}{int(names[i][-1])+240 // 15}")
            tmp_fields += _fields
            tmp_names += _names
        config["feature_15min"] = (tmp_fields, tmp_names)
        # label
        config["label"] = (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
        return config
