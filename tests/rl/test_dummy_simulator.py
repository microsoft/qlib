import pandas as pd

from qlib.backtest import Order
from qlib.rl.tasks.data import pickle_styled
from qlib.rl.tasks.order_execution import SingleAssetOrderExecution

from pathlib import Path


DATA_DIR = Path('/mnt/data/Sample-Testdata/us/')  # Update this when infrastructure is built.


def test_pickle_data_inspect():
    data = pickle_styled.get_intraday_backtest_data(DATA_DIR / 'raw', 'AAL', '2013-12-11', 'close', 0)
    assert len(data) == 390

    data = pickle_styled.get_intraday_processed_data(DATA_DIR / 'processed', 'AAL', '2013-12-11', 5, data.get_time_index())
    assert len(data.today) == len(data.yesterday) == 390


def test_simulator():
    order = Order(
        'AAL', 30., 0,
        pd.Timestamp('2013-12-11 00:00:00'),
        pd.Timestamp('2013-12-11 23:59:59')
    )

    simulator = SingleAssetOrderExecution(order, DATA_DIR / 'raw')
    state = simulator.get_state()
    print(state)

    # start_time = 15
    # end_time = 225
    # max_time = 300

    # state = SAOEEpisodicState(start_time, end_time, 30, None, arr_fn, arr_fn,
    #                           do_nothing, do_nothing, 1, 1000., 1000., FlowDirection.LIQUIDATE)
    # assert state.next_interval() == (0, 15)
    # state.step(np.random.uniform(0, 1, (state.next_duration(), )))
    # assert state.next_interval() == (15, 45)
    # for _ in range(6):
    #     state.step(np.random.uniform(0, 1e-3, (state.next_duration(), )))
    # assert state.next_interval() == (195, 210)
    # state.step(np.random.uniform(0, 1, (state.next_duration(), )))


test_pickle_data_inspect()
test_simulator()
