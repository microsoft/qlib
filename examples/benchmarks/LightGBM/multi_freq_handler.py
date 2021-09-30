#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pandas as pd

from qlib.data.dataset.loader import QlibDataLoader
from qlib.contrib.data.handler import DataHandlerLP, _DEFAULT_LEARN_PROCESSORS, check_transform_proc


class Avg15minLoader(QlibDataLoader):
    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        df = super(Avg15minLoader, self).load(instruments, start_time, end_time)
        if self.is_group:
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
        inst_processor=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        data_loader = Avg15minLoader(
            config=self.loader_config(), filter_pipe=filter_pipe, freq=freq, inst_processor=inst_processor
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
        fields = ["$close", "$open", "$low", "$high", "$volume", "$vwap"]
        names = list(map(lambda x: x.strip("$") + "0", fields))

        config = {"feature_day": (fields, names)}
        # features day
        # features 15min
        tmp_fields = []
        tmp_names = []
        # Ref(Mean($close, 15), 0), Ref(Mean($close, 15), 14)
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
