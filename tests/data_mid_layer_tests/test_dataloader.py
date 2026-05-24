# TODO:
# dump alpha 360 to dataframe and merge it with Alpha158

import sys
import unittest
import qlib
import pandas as pd
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parent))
from qlib.data.dataset.loader import NestedDataLoader, QlibDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from qlib.data.dataset.processor import Fillna
from qlib.data import D


class TestDataLoader(unittest.TestCase):

    def test_group_loader_applies_filter_pipe_once(self):
        filter_pipe = [{"filter_type": "NameDFilter", "name_rule_re": "SH.*"}]
        instruments = {
            "SH600000": [(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))]
        }
        loader = QlibDataLoader(
            config={
                "feature": (["$close"], ["CLOSE"]),
                "label": (["Ref($close, -1)"], ["LABEL0"]),
            },
            filter_pipe=filter_pipe,
            swap_level=False,
        )

        def mock_features(
            instruments_arg, exprs, start_time, end_time, freq, inst_processors
        ):
            self.assertIs(instruments_arg, instruments)
            return pd.DataFrame(
                [[1.0]], index=pd.Index(["SH600000"], name="instrument")
            )

        with patch(
            "qlib.data.dataset.loader.D.instruments", return_value=instruments
        ) as mock_instruments, patch(
            "qlib.data.dataset.loader.D.features", side_effect=mock_features
        ) as mock_features_fn:
            df = loader.load(
                instruments="csi300", start_time="2020-01-01", end_time="2020-01-02"
            )

        mock_instruments.assert_called_once_with("csi300", filter_pipe=filter_pipe)
        self.assertEqual(mock_features_fn.call_count, 2)
        self.assertEqual(list(df.columns.get_level_values(0)), ["feature", "label"])

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
