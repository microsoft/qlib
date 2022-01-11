import unittest
from .data import GetData
from .. import init
from ..constant import REG_CN
from qlib.data.filter import NameDFilter
from qlib.data import D
from qlib.data.data import Cal, DatasetD


class TestAutoData(unittest.TestCase):

    _setup_kwargs = {}
    provider_uri = "~/.qlib/qlib_data/cn_data_simple"  # target_dir
    provider_uri_1day = "~/.qlib/qlib_data/cn_data"  # target_dir
    provider_uri_1min = "~/.qlib/qlib_data/cn_data_1min"

    @classmethod
    def setUpClass(cls, enable_1d_type="simple", enable_1min=False) -> None:
        # use default data

        if enable_1d_type == "simple":
            provider_uri_day = cls.provider_uri
            name_day = "qlib_data_simple"
        elif enable_1d_type == "full":
            provider_uri_day = cls.provider_uri_1day
            name_day = "qlib_data"
        else:
            raise NotImplementedError(f"This type of input is not supported")

        GetData().qlib_data(
            name=name_day,
            region=REG_CN,
            interval="1d",
            target_dir=provider_uri_day,
            delete_old=False,
            exists_skip=True,
        )

        if enable_1min:
            GetData().qlib_data(
                name="qlib_data",
                region=REG_CN,
                interval="1min",
                target_dir=cls.provider_uri_1min,
                delete_old=False,
                exists_skip=True,
            )

        provider_uri_map = {"1min": cls.provider_uri_1min, "day": provider_uri_day}
        init(
            provider_uri=provider_uri_map,
            region=REG_CN,
            expression_cache=None,
            dataset_cache=None,
            **cls._setup_kwargs,
        )


class TestOperatorData(TestAutoData):
    @classmethod
    def setUpClass(cls, enable_1d_type="simple", enable_1min=False) -> None:
        # use default data
        super().setUpClass(enable_1d_type, enable_1min)
        nameDFilter = NameDFilter(name_rule_re="SH600110")
        instruments = D.instruments("csi300", filter_pipe=[nameDFilter])
        start_time = "2005-01-04"
        end_time = "2005-12-31"
        freq = "day"

        instruments_d = DatasetD.get_instruments_d(instruments, freq)
        cls.instruments_d = instruments_d
        cal = Cal.calendar(start_time, end_time, freq)
        cls.cal = cal
        cls.start_time = cal[0]
        cls.end_time = cal[-1]
        cls.inst = list(instruments_d.keys())[0]
        cls.spans = list(instruments_d.values())[0]
