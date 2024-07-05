# TODO:
# dump alpha 360 to dataframe and merge it with Alpha158

import sys
import unittest
import qlib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from qlib.data.dataset.loader import NestedDataLoader
from qlib.contrib.data.loader import Alpha158DL, Alpha360DL


class TestDataLoader(unittest.TestCase):

    def test_nested_data_loader(self):
        qlib.init()
        nd = NestedDataLoader(
            dataloader_l=[
                {
                    "class": "qlib.contrib.data.loader.Alpha158DL",
                },
                {
                    "class": "qlib.contrib.data.loader.Alpha360DL",
                    "kwargs": {"config": {"label": (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])}},
                },
            ]
        )
        # Of course you can use StaticDataLoader

        dataset = nd.load()

        assert dataset is not None

        columns = dataset.columns.tolist()
        columns_list = [tup[1] for tup in columns]

        for col in Alpha158DL.get_feature_config()[1]:
            assert col in columns_list

        for col in Alpha360DL.get_feature_config()[1]:
            assert col in columns_list

        # 断言标签也包含在数据中
        assert "LABEL0" in columns_list

        # Then you can use it wth DataHandler;


if __name__ == "__main__":
    unittest.main()
