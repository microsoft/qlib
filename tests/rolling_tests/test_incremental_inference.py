import pandas as pd
import pytest

import qlib.workflow.online.update as update
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow.online.update import (
    get_incremental_inference_config,
    prepare_incremental_inference_dataset,
)


class DummyDataset:
    def __init__(self):
        self.config_kwargs = None
        self.setup_kwargs = None

    def config(self, **kwargs):
        self.config_kwargs = kwargs

    def setup_data(self, **kwargs):
        self.setup_kwargs = kwargs


def test_get_incremental_inference_config_extends_only_loading_window(monkeypatch):
    def fake_get_date_by_shift(trading_date, shift, clip_shift=True, freq="day"):
        assert pd.Timestamp(trading_date) == pd.Timestamp("2020-01-03")
        assert shift == -2
        assert clip_shift is False
        assert freq == "day"
        return pd.Timestamp("2020-01-01")

    monkeypatch.setattr(update, "get_date_by_shift", fake_get_date_by_shift)

    data_start_time, segments = get_incremental_inference_config(
        start_time="2020-01-03",
        end_time="2020-01-05",
        hist_ref=2,
    )

    assert data_start_time == pd.Timestamp("2020-01-01")
    assert segments == {
        "test": (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-05"))
    }


def test_prepare_incremental_inference_dataset_configures_dataset(monkeypatch):
    monkeypatch.setattr(
        update,
        "get_date_by_shift",
        lambda trading_date, shift, clip_shift=True, freq="day": pd.Timestamp(
            "2020-01-01"
        ),
    )
    dataset = DummyDataset()

    result = prepare_incremental_inference_dataset(
        dataset,
        start_time="2020-01-03",
        end_time="2020-01-05",
        hist_ref=2,
    )

    assert result is dataset
    assert dataset.config_kwargs == {
        "handler_kwargs": {
            "start_time": pd.Timestamp("2020-01-01"),
            "end_time": pd.Timestamp("2020-01-05"),
        },
        "segments": {
            "test": (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-05"))
        },
    }
    assert dataset.setup_kwargs == {"handler_kwargs": {"init_type": DataHandlerLP.IT_LS}}


def test_prepare_incremental_inference_dataset_infers_ts_hist_ref(monkeypatch):
    class FakeTSDatasetH(DummyDataset):
        step_len = 4

    def fake_get_date_by_shift(trading_date, shift, clip_shift=True, freq="day"):
        assert shift == -3
        return pd.Timestamp("2020-01-01")

    monkeypatch.setattr(update, "TSDatasetH", FakeTSDatasetH)
    monkeypatch.setattr(update, "get_date_by_shift", fake_get_date_by_shift)
    dataset = FakeTSDatasetH()

    prepare_incremental_inference_dataset(
        dataset,
        start_time="2020-01-04",
        end_time="2020-01-05",
    )

    assert dataset.config_kwargs["handler_kwargs"]["start_time"] == pd.Timestamp(
        "2020-01-01"
    )
    assert dataset.config_kwargs["segments"] == {
        "test": (pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-05"))
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_time": "2020-01-05", "end_time": "2020-01-03"},
        {"start_time": "2020-01-03", "end_time": "2020-01-05", "hist_ref": -1},
        {
            "start_time": "2020-01-03",
            "end_time": "2020-01-05",
            "segment_name": "",
        },
    ],
)
def test_get_incremental_inference_config_validates_inputs(kwargs):
    with pytest.raises(ValueError):
        get_incremental_inference_config(**kwargs)
