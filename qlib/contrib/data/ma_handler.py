# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader


class MADataLoader(QlibDataLoader):
    """Dataloader to get MA factors"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        """create factors for MA strategy"""
        fields = []
        names = []
        
        # 计算短期移动平均线 (MA5)
        fields += ["Mean($close, 5)/$close"]
        names += ["MA5"]
        
        # 计算长期移动平均线 (MA20)
        fields += ["Mean($close, 20)/$close"]
        names += ["MA20"]
        
        # 计算均线差
        fields += ["(Mean($close, 5) - Mean($close, 20))/$close"]
        names += ["MA_DIFF"]
        
        # 计算均线交叉信号
        fields += ["If(Mean($close, 5)/$close > Mean($close, 20)/$close, 1, -1)"]
        names += ["MA_SIGNAL"]
        
        return fields, names


class MAHandler(DataHandlerLP):
    """Data handler for MA strategy"""

    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        data_loader = {
            "class": "MADataLoader",
            "kwargs": {
                "config": {
                    "feature": MADataLoader.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
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
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
