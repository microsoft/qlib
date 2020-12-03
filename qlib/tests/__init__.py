import sys
import unittest
from ..utils import exists_qlib_data
from .data import GetData
from .. import init
from ..config import REG_CN


class TestAutoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # use default data
        provider_uri = "~/.qlib/qlib_data/cn_data_simple"  # target_dir
        if not exists_qlib_data(provider_uri):
            print(f"Qlib data is not found in {provider_uri}")

            GetData().qlib_data(
                name="qlib_data_simple", region="cn", version="latest", interval="1d", target_dir=provider_uri
            )
        init(provider_uri=provider_uri, region=REG_CN)
