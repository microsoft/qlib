from qlib.rl.tasks.data.pickle_styled import get_intraday_backtest_data

from pathlib import Path


DATA_DIR = Path('/mnt/data/Sample-Testdata/us/')  # Update this when infrastructure is built.


def test_pickle_data_inspect():
    data = get_intraday_backtest_data(DATA_DIR / 'raw' / 'AAL.pkl.backtest', 'AAL', '2013-12-11')
    print(data)
