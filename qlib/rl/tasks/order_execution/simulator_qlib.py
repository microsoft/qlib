import gym
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional, Any, Union, Dict, Callable, Tuple

from qlib.backtest.decision import BaseTradeDecision, Order, OrderHelper, TradeDecisionWO, TradeRangeByTime, TradeRange
from qlib.backtest.executor import BaseExecutor
from qlib.backtest.utils import CommonInfrastructure, TradeCalendarManager, get_start_end_idx
from qlib.strategy.base import BaseStrategy
from torch.utils.data import Dataset

from .simulator_simple import SingleAssetOrderExecutionState

from qlib.rl.simulator import Simulator
from qlib.integration import QlibConfig, ExchangeConfig


from collections import defaultdict

from typing import Any, List, Literal, Optional, Tuple, overload, Union

import pandas as pd
import qlib.contrib.strategy
from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.strategy.base import BaseStrategy
from tianshou.data import Batch
from tianshou.policy import BasePolicy

from .infrastructure import StateMaintainer
from .predictor import MultiplexPredictor




class SingleOrderStrategy(BaseStrategy):
    # this logic is copied from FileOrderStrategy
    def __init__(self, common_infra: CommonInfrastructure, order: Dict[str, Any], trade_range: TradeRange):
        super().__init__(common_infra=common_infra)
        self.order = order
        self.trade_range = trade_range

    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        oh: OrderHelper = self.common_infra.get("trade_exchange").get_order_helper()
        order_list = [oh.create(
            code=self.order['instrument'],
            amount=self.order['amount'],
            direction=Order.parse_dir(self.order['direction'])
        )]
        trade_decision = TradeDecisionWO(order_list, self, self.trade_range)
        return trade_decision


class SingleAssetOrderExecutionQlib(Simulator[Order, SingleAssetOrderExecutionState]):
    def __init__(
        self,
        time_per_step: str,  # e.g., 30min
        start_time: str,  # e.g., 9:30
        end_time: str,  # e.g., 14:57
        qlib_config: QlibConfig,
        exchange_config: ExchangeConfig,
        action: Optional[BaseAction] = None,
        observation: Optional[BaseObservation] = None,
        reward: Optional[BaseReward] = None,
        dataloader: Optional[Iterable[Dict[str, Any]]] = None,
        inner_executor_fn: Union[Dict[str, SubclassConfig[BaseStrategy]],
                                 Callable[[CommonInfrastructure], BaseExecutor], None] = None,
    ):
        self.time_per_step = time_per_step
        self.trade_range = TradeRangeByTime(start_time, end_time)
        self.qlib_config = qlib_config
        self.exchange_config = exchange_config
        self.inner_strategy = DecomposedStrategy()  # empty strategy
        self.action = action
        self.observation = observation
        self.reward = reward
        self.dataloader = dataloader

        if isinstance(inner_executor_fn, dict):
            self.inner_executor_fn = lambda common_infra: get_multi_level_executor(
                common_infra, self.exchange_config, inner_executor_fn)
        else:
            self.inner_executor_fn = inner_executor_fn

    @property
    def action_space(self):
        return self.action.action_space

    @property
    def observation_space(self):
        return self.observation.observation_space

    def _reset(self, order: Dict[str, Any]):
        # make qlib aware that now it's operate on a specific data
        init_qlib(self.qlib_config, order['instrument'])
        common_infra = get_common_infra(
            self.exchange_config,
            trade_start_time=order['datetime'],
            trade_end_time=order['datetime'],
            codes=[order['instrument']],  # ignore cash limit for now
        )

        # e.g., for 1-day-to-30-min order split, the executor here is the one that interacts with the strategy
        # that we wish to optimize
        inner_executor = self.inner_executor_fn(common_infra)
        # we run `collect_data` only once, and one step is one day
        # the executor here is the executor with the "loop"
        # the strategy we want to optimize is the inner strategy of this executor
        self.executor = RLNestedExecutor('1day', inner_executor, self.inner_strategy,
                                         common_infra=common_infra, track_data=True)

        # mock a top-level strategy to get a valid trade decision
        top_strategy = SingleOrderStrategy(common_infra, order, self.trade_range)

        self.executor.reset(start_time=order['datetime'], end_time=order['datetime'])
        top_strategy.reset(level_infra=self.executor.get_level_infra())
        trade_decision = top_strategy.generate_trade_decision()

        self.collect_data_loop = self.executor.collect_data(trade_decision, level=0)

        strategy = next(self.collect_data_loop)
        while not isinstance(strategy, BaseStrategy):
            strategy = next(self.collect_data_loop)

        assert isinstance(strategy, DecomposedStrategy)
        sample, ep_state = strategy.sample_state_pair

        # history
        self.action_history = np.full(ep_state.num_step, np.nan)
        self.prev_ep_state = ep_state

        return self.observation(sample, ep_state)

    def reset(self) -> Optional[BaseObservation]:
        try:
            cur_order = next(self.dataloader)
            return self._reset(cur_order)
        except StopIteration:
            self.dataloader = None
            return generate_nan_observation(self.observation_space)

    def step(self, action: Any):
        assert self.dataloader is not None
        # Action is what we have got from policy
        self.action_history[self.prev_ep_state.cur_step] = action
        action = self.action(action, self.prev_ep_state)

        try:
            # to get next observation
            # could raise an stop iteration error here
            strategy = self.collect_data_loop.send(action)
            while not isinstance(strategy, BaseStrategy):
                strategy = self.collect_data_loop.send(action)
            assert isinstance(strategy, DecomposedStrategy)
            sample, ep_state = strategy.sample_state_pair
            state_start, state_end = ep_state.last_interval

            # FIXME: this is a temporary patch to make qlib integration compatible with old design. This callback
            # function should be pass to state directly during initialization or set afterwards rather than directly
            # called here.
            self.reward.step_end_callback(state_start, state_end, ep_state)

        except StopIteration:
            # done
            sample, ep_state = self.inner_strategy.sample_state_pair
            assert ep_state.done
            self.reward.episode_end_callback(ep_state)

        self.prev_ep_state = ep_state

        reward, rew_info = self.reward(ep_state)

        info = {"category": ep_state.flow_dir.value, "reward": rew_info}
        if ep_state.done:
            info["index"] = {"stock_id": sample.stock_id, "date": sample.date}
            info["history"] = {"action": self.action_history}
            info.update(ep_state.logs())

            try:
                # done but loop is not exhausted
                # exhaust the loop manually
                while True:
                    self.collect_data_loop.send(0.)
            except StopIteration:
                pass

            info["qlib"] = {}
            for key, val in list(
                self.executor.trade_account.get_trade_indicator().order_indicator_his.values()
            )[0].to_series().items():
                info["qlib"][key] = val.item()

        return self.observation(sample, ep_state), reward, ep_state.done, info


def read_order_file(order_file: Union[Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(order_file, pd.DataFrame):
        return order_file

    order_file = Path(order_file)

    if order_file.suffix == '.pkl':
        order_df = pd.read_pickle(order_file).reset_index()
    elif order_file.suffix == '.csv':
        order_df = pd.read_csv(order_file)
    else:
        raise TypeError(f'Unsupported order file type: {order_file}')

    if 'date' in order_df.columns:
        # legacy dataframe columns
        order_df = order_df.rename(columns={'date': 'datetime', 'order_type': 'direction'})
    order_df['datetime'] = order_df['datetime'].astype(str)

    return order_df


@DATASETS.register_module("intradaysa_qlib")
class OrderDataset(Dataset):
    def __init__(self, order_file: Path):
        self.order_df = pd.read_csv(order_file)

    def __getitem__(self, idx):
        return self.order_df.iloc[idx].to_dict()

    def __len__(self):
        return len(self.order_df)
