import pandas as pd

from qlib.backtest.decision import Order, OrderDir
from qlib.rl.order_execution import CategoricalActionInterpreter
from qlib.rl.order_execution.tests.common import get_simulator


def is_close(a: float, b: float, epsilon: float = 1e-4) -> bool:
    return abs(a - b) <= epsilon


def test_simulator_first_step():
    TOTAL_POSITION = 2100.0

    order = Order(
        stock_id="SH600000",
        amount=TOTAL_POSITION,
        direction=OrderDir.BUY,
        start_time=pd.Timestamp("2019-03-04 09:30:00"),
        end_time=pd.Timestamp("2019-03-04 14:29:00"),
    )

    simulator = get_simulator(order)
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp('2019-03-04 09:30:00')
    assert state.position == TOTAL_POSITION

    AMOUNT = 300.0
    simulator.step(AMOUNT)
    state = simulator.get_state()
    assert state.cur_time == pd.Timestamp('2019-03-04 10:00:00')
    assert state.position == TOTAL_POSITION - AMOUNT
    assert len(state.history_exec) == 30
    assert state.history_exec.index[0] == pd.Timestamp('2019-03-04 09:30:00')

    assert is_close(state.history_exec["market_volume"].iloc[0], 109382.382812)
    assert is_close(state.history_exec["market_price"].iloc[0], 149.566483)
    assert (state.history_exec["amount"] == AMOUNT / 30).all()
    assert (state.history_exec["deal_amount"] == AMOUNT / 30).all()
    assert is_close(state.history_exec["trade_price"].iloc[0], 149.566483)
    assert is_close(state.history_exec["trade_value"].iloc[0], 1495.664825)
    assert is_close(state.history_exec["position"].iloc[0], TOTAL_POSITION - AMOUNT / 30)
    # assert state.history_exec["ffr"].iloc[0] == 1 / 60  # FIXME

    assert is_close(state.history_steps["market_volume"].iloc[0], 1254848.5756835938)
    assert state.history_steps["amount"].iloc[0] == AMOUNT
    assert state.history_steps["deal_amount"].iloc[0] == AMOUNT
    assert state.history_steps["ffr"].iloc[0] == 1.0
    assert is_close(
        state.history_steps["pa"].iloc[0] * (1.0 if order.direction == OrderDir.SELL else -1.0),
        (state.history_steps["trade_price"].iloc[0] / simulator.twap_price - 1) * 10000,
    )


def test_simulator_stop_twap() -> None:
    TOTAL_POSITION = 2100.0

    order = Order(
        stock_id="SH600000",
        amount=TOTAL_POSITION,
        direction=OrderDir.BUY,
        start_time=pd.Timestamp("2019-03-04 09:30:00"),
        end_time=pd.Timestamp("2019-03-04 14:29:00"),
    )

    simulator = get_simulator(order)
    NUM_STEPS = 7
    for i in range(NUM_STEPS):
        simulator.step(TOTAL_POSITION / NUM_STEPS)

    HISTORY_STEP_LENGTH = 30 * NUM_STEPS
    state = simulator.get_state()
    assert len(state.history_exec) == HISTORY_STEP_LENGTH

    assert (state.history_exec["deal_amount"] == TOTAL_POSITION / HISTORY_STEP_LENGTH).all()
    assert is_close(state.history_steps["position"].iloc[0], TOTAL_POSITION * (NUM_STEPS - 1) / NUM_STEPS)
    assert is_close(state.history_steps["position"].iloc[-1], 0.0)
    assert is_close(state.position, 0.0)
    assert is_close(state.metrics["ffr"], 1.0)

    assert is_close(state.metrics["market_price"], state.backtest_data.get_deal_price().mean())
    assert is_close(state.metrics["market_volume"], state.backtest_data.get_volume().sum())
    assert is_close(state.metrics["trade_price"], state.metrics["market_price"])
    assert is_close(state.metrics["pa"], 0.0)


if __name__ == "__main__":
    test_simulator_first_step()
    test_simulator_stop_twap()
