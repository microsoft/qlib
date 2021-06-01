import unittest
from .data import GetData
from .. import init
from ..config import REG_CN


class TestAutoData(unittest.TestCase):

    _setup_kwargs = {}
    provider_uri = "~/.qlib/qlib_data/cn_data_simple"  # target_dir

    @classmethod
    def setUpClass(cls) -> None:
        # use default data

        GetData().qlib_data(
            name="qlib_data_simple",
            region=REG_CN,
            interval="1d",
            target_dir=cls.provider_uri,
            delete_old=False,
            exists_skip=True,
        )
        init(provider_uri=cls.provider_uri, region=REG_CN, **cls._setup_kwargs)
