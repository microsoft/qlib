import sys
import unittest
from ..utils import exists_qlib_data
from .data import GetData
from .. import init
from ..config import REG_CN


class TestAutoData(unittest.TestCase):

    _setup_kwargs = {}
    provider_uri = "~/.qlib/qlib_data/cn_data_simple"  # target_dir

    @classmethod
    def setUpClass(cls) -> None:
        # use default data
        if not exists_qlib_data(cls.provider_uri):
            print(f"Qlib data is not found in {cls.provider_uri}")

            GetData().qlib_data(
                name="qlib_data_simple",
                region="cn",
                interval="1d",
                target_dir=cls.provider_uri,
                delete_old=False,
            )
        init(provider_uri=cls.provider_uri, region=REG_CN, **cls._setup_kwargs)
