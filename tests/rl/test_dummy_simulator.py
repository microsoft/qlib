from qlib.rl.tasks.data import pickle_styled

from pathlib import Path


DATA_DIR = Path('/mnt/data/Sample-Testdata/us/')  # Update this when infrastructure is built.


def test_pickle_data_inspect():
    data = pickle_styled.get_intraday_backtest_data(DATA_DIR / 'raw', 'AAL', '2013-12-11', 'close')
    print(data)

    data = pickle_styled.get_intraday_processed_data(DATA_DIR / 'processed', 'AAL', '2013-12-11', 5, data.get_time_index())
    print(data)

test_pickle_data_inspect()
