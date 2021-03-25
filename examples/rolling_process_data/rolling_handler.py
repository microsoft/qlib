from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import DataLoaderDH
from qlib.contrib.data.handler import check_transform_proc


class RollingDataHandler(DataHandlerLP):
    def __init__(
        self,
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        data_loader_kwargs={},
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "DataLoaderDH",
            "kwargs": {**data_loader_kwargs},
        }

        super().__init__(
            instruments=None,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )
