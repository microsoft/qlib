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
from qlib.contrib.data.handler import Alpha158
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

    def test_grouped_loader_applies_filter_pipe_once(self):
        loader = QlibDataLoader(
            config={
                "feature": (["$close"], ["close"]),
                "label": (["Ref($close, -1)"], ["label"]),
            },
            filter_pipe=[{"filter_type": "SeriesDFilter"}],
        )
        instruments_config = {"market": "csi300", "filter_pipe": loader.filter_pipe}
        index = pd.MultiIndex.from_tuples(
            [("SH600000", pd.Timestamp("2020-01-01"))],
            names=["instrument", "datetime"],
        )

        def fake_features(instruments, exprs, start_time, end_time, freq="day", inst_processors=None):
            self.assertIs(instruments, instruments_config)
            return pd.DataFrame([[1.0] * len(exprs)], index=index, columns=exprs)

        with (
            patch(
                "qlib.data.dataset.loader.D.instruments",
                return_value=instruments_config,
                create=True,
            ) as instruments_mock,
            patch("qlib.data.dataset.loader.D.features", side_effect=fake_features, create=True) as features_mock,
        ):
            df = loader.load("csi300", start_time="2020-01-01", end_time="2020-01-02")

        instruments_mock.assert_called_once_with("csi300", filter_pipe=loader.filter_pipe)
        self.assertEqual(features_mock.call_count, 2)
        self.assertEqual({"feature", "label"}, set(df.columns.get_level_values(0)))

    def test_alpha158_handler_applies_filter_pipe_once(self):
        filter_pipe = [{"filter_type": "SeriesDFilter"}]
        instruments_config = {"market": "csi300", "filter_pipe": filter_pipe}
        index = pd.MultiIndex.from_tuples(
            [("SH600000", pd.Timestamp("2020-01-01"))],
            names=["instrument", "datetime"],
        )

        def fake_features(instruments, exprs, start_time, end_time, freq="day", inst_processors=None):
            self.assertIs(instruments, instruments_config)
            return pd.DataFrame([[1.0] * len(exprs)], index=index, columns=exprs)

        with (
            patch(
                "qlib.data.dataset.loader.D.instruments",
                return_value=instruments_config,
                create=True,
            ) as instruments_mock,
            patch("qlib.data.dataset.loader.D.features", side_effect=fake_features, create=True) as features_mock,
        ):
            handler = Alpha158(
                instruments="csi300",
                start_time="2020-01-01",
                end_time="2020-01-02",
                filter_pipe=filter_pipe,
                infer_processors=[],
                learn_processors=[],
            )

        instruments_mock.assert_called_once_with("csi300", filter_pipe=filter_pipe)
        self.assertEqual(features_mock.call_count, 2)
        self.assertEqual({"feature", "label"}, set(handler._data.columns.get_level_values(0)))


if __name__ == "__main__":
    unittest.main()
