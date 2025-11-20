# TODO:
# dump alpha 360 to dataframe and merge it with Alpha158

import sys
import unittest
import qlib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from qlib.data.dataset.loader import NestedDataLoader, QlibDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from qlib.data.dataset.processor import Fillna
from qlib.data import D


class TestDataLoader(unittest.TestCase):

    def test_nested_data_loader(self):
        qlib.init(kernels=1)
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

        dataset = nd.load(instruments="csi300", start_time="2020-01-01", end_time="2020-01-31")

        assert dataset is not None

        columns = dataset.columns.tolist()
        columns_list = [tup[1] for tup in columns]

        for col in Alpha158DL.get_feature_config()[1]:
            assert col in columns_list

        for col in Alpha360DL.get_feature_config()[1]:
            assert col in columns_list

        assert "LABEL0" in columns_list

        assert dataset.isna().any().any()

        fn = Fillna(fields_group="feature", fill_value=0)
        fn_dataset = fn.__call__(dataset)

        assert not fn_dataset.isna().any().any()

        # Then you can use it wth DataHandler;
        # NOTE: please note that the data processors are missing!!!  You should add based on your requirements

        """
        dataset.to_pickle("test_df.pkl")
        nested_data_loader = NestedDataLoader(
            dataloader_l=[
                {
                    "class": "qlib.contrib.data.loader.Alpha158DL",
                    "kwargs": {"config": {"label": (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])}},
                },
                {
                    "class": "qlib.contrib.data.loader.Alpha360DL",
                },
                {
                    "class": "qlib.data.dataset.loader.StaticDataLoader",
                    "kwargs": {"config": "test_df.pkl"},
                },
            ]
        )
        data_handler_config = {
            "start_time": "2008-01-01",
            "end_time": "2020-08-01",
            "instruments": "csi300",
            "data_loader": nested_data_loader,
        }
        data_handler = DataHandlerLP(**data_handler_config)
        data = data_handler.fetch()
        print(data)
        """


if __name__ == "__main__":
    unittest.main()
