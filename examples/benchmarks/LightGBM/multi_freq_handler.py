#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import warnings
from pathlib import Path
from typing import Union
import pandas as pd

from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.contrib.data.handler import DataHandlerLP, _DEFAULT_LEARN_PROCESSORS, check_transform_proc


class MultiFreqLoader(QlibDataLoader):
    def load_group_df(
        self,
        instruments,
        exprs: list,
        names: list,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        gp_name: str = None,
    ) -> pd.DataFrame:
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            instruments = "all"
        if isinstance(instruments, str):
            instruments = D.instruments(instruments, filter_pipe=self.filter_pipe)
        elif self.filter_pipe is not None:
            warnings.warn("`filter_pipe` is not None, but it will not be used with `instruments` as list")

        if gp_name == "feature":
            # freq == day
            _exps = list(filter(lambda x: not x.startswith("Avg"), exprs))
            _day_df = D.features(instruments, _exps, start_time, end_time, freq="day")
            _day_df.columns = list(filter(lambda x: int("".join(filter(str.isdigit, x))) == 0, names))
            # freq == 1min
            _exps = list(filter(lambda x: x.startswith("Avg"), exprs))
            _min_df = D.features(
                instruments,
                _exps,
                start_time,
                end_time,
                freq="1min",
                inst_processors=self.inst_processor.get("feature", []),
            )
            _min_df.columns = list(filter(lambda x: int("".join(filter(str.isdigit, x))) > 0, names))
            df = pd.concat([_day_df, _min_df], axis=1, sort=False)
        elif gp_name == "label":
            freq = self.freq[gp_name] if isinstance(self.freq, dict) else self.freq
            df = D.features(
                instruments,
                exprs,
                start_time,
                end_time,
                freq=freq,
                inst_processors=self.inst_processor.get(gp_name, []),
            )
            df.columns = names
        else:
            raise ValueError(f"Unsupported gp_name: {gp_name}")

        if self.swap_level:
            df = df.swaplevel().sort_index()  # NOTE: if swaplevel, return <datetime, instrument>
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

        data_loader = {
            "class": "MultiFreqLoader",
            "module_path": str(Path(__file__).resolve()),
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.get("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def get_feature_config(self):
        fields = ["$close", "$open", "$low", "$high", "$volume", "$vwap"]
        names = list(map(lambda x: x.strip("$") + "0", fields))
        tmp_fields = []
        tmp_names = []
        for i, _f in enumerate(fields):
            _fields = [f"Avg({_f}, {15 * j}, {15 * j + 15}, 'nanmean')" for j in range(0, 240 // 15)]
            _names = [f"{names[i][:-1]}{int(names[i][-1])+j}" for j in range(1, 240 // 15 + 1)]
            tmp_fields += _fields
            tmp_names += _names
        return fields + tmp_fields, names + tmp_names

    def get_label_config(self):
        return (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
