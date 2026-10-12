import pandas as pd

from qlib.contrib.model.pytorch_gats_ts import DailyBatchSampler
from qlib.data.dataset import TSDataSampler


def test_daily_batch_sampler_groups_actual_positions_by_datetime():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    instruments = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    data = pd.DataFrame({"feature": range(len(index)), "label": range(len(index))}, index=index)
    data_source = TSDataSampler(data, dates[0], dates[-1], step_len=2)

    sampler = DailyBatchSampler(data_source)
    source_index = data_source.get_index()
    batches = list(sampler)

    assert len(sampler) == len(dates)
    for date, batch in zip(dates, batches):
        batch_index = source_index[batch]
        assert batch_index.get_level_values("datetime").unique().tolist() == [date]
        assert batch_index.get_level_values("instrument").tolist() == instruments

    expected_index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    assert sampler.get_index().equals(expected_index)
