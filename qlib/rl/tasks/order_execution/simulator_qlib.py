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


from collections import defaultdict

from typing import Any, List, Literal, Optional, Tuple, overload, Union

import pandas as pd
import qlib.contrib.strategy
from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.strategy.base import BaseStrategy
from tianshou.data import Batch
from tianshou.policy import BasePolicy
from utilsd.config import SubclassConfig

from .infrastructure import StateMaintainer
from .predictor import MultiplexPredictor


class TradeDecisionWithDetails(TradeDecisionWO):
    def __init__(self, order_list: List[Order], strategy: BaseStrategy,
                 trade_range: Optional[Tuple[int, int]] = None, details: Optional[Any] = None):
        super().__init__(order_list, strategy, trade_range)
        self.details = details


class TWAPStrategy(qlib.contrib.strategy.TWAPStrategy):
    def __init__(self):
        super().__init__()


class RLStrategyBase(BaseStrategy):
    def post_exe_step(self, execute_result):
        """
        post process for each step of strategy this is design for RL Strategy,
        which require to update the policy state after each step

        NOTE: it is strongly coupled with RLNestedExecutor;
        """
        raise NotImplementedError("Please implement the `post_exe_step` method")


class RLStrategy(RLStrategyBase):
    """Qlib RL strategy that wraps neutrader components."""

    @overload
    def __init__(self, observation: BaseObservation, action: BaseAction, policy: Optional[BasePolicy]):
        ...

    @overload
    def __init__(self, observation: SubclassConfig[BaseObservation],
                 action: SubclassConfig[BaseAction], policy: Optional[SubclassConfig[BasePolicy]],
                 network: SubclassConfig[BaseNetwork]):
        ...

    def __init__(self,
                 observation: Union[BaseObservation, SubclassConfig[BaseObservation]],
                 action: Union[BaseAction, SubclassConfig[BaseAction]],
                 policy: Union[BasePolicy, SubclassConfig[BasePolicy]],
                 network: Optional[SubclassConfig[BaseNetwork]] = None):
        super().__init__()
        if observation is None:
            self.observation = self.action = self.policy = None
        elif isinstance(observation, BaseObservation):
            self.observation = observation
            self.action = action
            self.policy = policy
        else:
            self.observation = observation.build()
            self.action = action.build()

            if network is not None:
                network = network.build()
                if policy is not None:
                    self.policy = policy.build(network=network,
                                               obs_space=self.observation.observation_space,
                                               action_space=self.action.action_space)
            else:
                if policy is not None:
                    policy = policy.build(obs_space=self.observation.observation_space,
                                          action_space=self.action.action_space)
        if self.policy is not None:
            self.policy.eval()

        self.maintainer: Optional[StateMaintainer] = None

    def reset(self, outer_trade_decision=None, **kwargs):
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)
        time_per_step = int(pd.Timedelta(self.trade_calendar.get_freq()) / pd.Timedelta('1min'))
        if outer_trade_decision is not None:
            self.maintainer = StateMaintainer(
                time_per_step,
                self.trade_calendar.get_all_time()[0],
                self.get_data_cal_avail_range(),
                self.trade_calendar.get_trade_step(),
                outer_trade_decision,
                self.trade_exchange
            )

    def post_exe_step(self, execute_result):
        self.maintainer.send_execute_result(execute_result)

    def generate_trade_details(self, act, exec_vols):
        trade_details = []
        for a, v, o in zip(act, exec_vols, self.outer_trade_decision.order_list):
            trade_details.append({
                'instrument': o.stock_id,
                'datetime': self.trade_calendar.get_step_time()[0],
                'freq': self.trade_calendar.get_freq(),
                'rl_exec_vol': v
            })
            if a is not None:
                trade_details[-1]['rl_action'] = a
        return pd.DataFrame.from_records(trade_details)

    def generate_trade_decision(self, execute_result=None):
        if self.outer_trade_decision is None:
            return TradeDecisionWO([], self)

        import torch  # to check tensor type
        obs_batch = []
        for sample, state in zip(self.maintainer.samples.values(), self.maintainer.states.values()):
            obs_batch.append({'obs': self.observation(sample, state)})
        with torch.no_grad():
            policy_out = self.policy(Batch(obs_batch))
        act = policy_out.act.numpy() if torch.is_tensor(policy_out.act) else policy_out.act
        exec_vols = [self.action(a, s) for a, s in zip(act, self.maintainer.states.values())]

        order_list = self.maintainer.generate_orders(self.get_data_cal_avail_range(rtype='step'), exec_vols)

        # NOTE: user don't have to care about index_range for each layer now.
        details = pd.DataFrame.from_records(self.generate_trade_details(act, exec_vols))
        return TradeDecisionWithDetails(order_list, self, details=details)


class MultiplexStrategyBase(RLStrategyBase):
    @overload
    def __init__(self, strategies: List[BaseStrategy]):
        ...

    @overload
    def __init__(self, strategies: List[SubclassConfig[BaseStrategy]]):
        ...

    def __init__(self, strategies: Union[List[BaseStrategy], List[SubclassConfig[BaseStrategy]]]):
        super().__init__()
        if isinstance(strategies[0], BaseStrategy):
            self.strategies = strategies
        else:
            self.strategies = [strategy.build() for strategy in strategies]

    def post_exe_step(self, execute_result):
        """
        post process for each step of strategy
        this is design for RL Strategy, which require to update the policy state after each step
        Multiplex Strategy may contains RL strategy, so it is responsible to update the state
        """
        raise NotImplementedError("Please implement the `post_exe_step` method")


class MultiplexStrategyOnTradeStep(MultiplexStrategyBase):
    """To use different strategy on different step of the outer calendar"""

    def reset_level_infra(self, level_infra):
        for strategy in self.strategies:
            strategy.reset_level_infra(level_infra)

    def reset_common_infra(self, common_infra):
        for strategy in self.strategies:
            strategy.reset_common_infra(common_infra)

    def reset(self, outer_trade_decision=None, **kwargs):
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)
        if outer_trade_decision is not None:
            outer_calendar = self.outer_trade_decision.strategy.trade_calendar
            self.strategies[outer_calendar.get_trade_step()].reset(outer_trade_decision=outer_trade_decision, **kwargs)

    def generate_trade_decision(self, execute_result=None):
        if self.outer_trade_decision is not None:
            outer_calendar = self.outer_trade_decision.strategy.trade_calendar
            return self.strategies[outer_calendar.get_trade_step()] \
                .generate_trade_decision(execute_result=execute_result)

    def post_exe_step(self, execute_result):
        if self.outer_trade_decision is not None:
            outer_calendar = self.outer_trade_decision.strategy.trade_calendar
            strategy = self.strategies[outer_calendar.get_trade_step()]
            if isinstance(strategy, RLStrategy):
                strategy.post_exe_step(execute_result=execute_result)


class MultiplexStrategyWithPredictor(MultiplexStrategyBase):
    """
    To use different strategy based on the choice of predictor.

    Predictor accepts a qlib Order and current time index (e.g., 0-239), returns an integer (e.g., 0-2).
    Ensemble weight should also be supported, e.g., [0.1, 0.7, 0.2], which will ensemble the decisions (order amount)
    with the weight.
    """

    @overload
    def __init__(self, strategies: List[BaseStrategy], predictor: MultiplexPredictor, on_reset: bool = False):
        ...

    @overload
    def __init__(self, strategies: List[SubclassConfig[BaseStrategy]],
                 predictor: SubclassConfig[MultiplexPredictor], on_reset: bool = False):
        ...

    def __init__(self,
                 strategies: Union[List[BaseStrategy], List[SubclassConfig[BaseStrategy]]],
                 predictor: Union[MultiplexPredictor, SubclassConfig[MultiplexPredictor]],
                 on_reset: bool = False):
        super().__init__(strategies=strategies)
        if isinstance(predictor, MultiplexPredictor):
            self.predictor = predictor
        else:
            self.predictor = predictor.build()
        self.on_reset = on_reset

    def reset_level_infra(self, level_infra):
        super().reset_level_infra(level_infra)
        for strategy in self.strategies:
            strategy.reset_level_infra(level_infra)

    def reset_common_infra(self, common_infra):
        super().reset_common_infra(common_infra)
        for strategy in self.strategies:
            strategy.reset_common_infra(common_infra)

    def reset(self, outer_trade_decision=None, **kwargs):
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)
        self.judge_cache = {}
        if outer_trade_decision is not None:
            if self.on_reset:
                for order in outer_trade_decision.order_list:
                    self.judge_cache[order.stock_id, order.direction] = self.predictor(order, 0)
            for strategy in self.strategies:
                strategy.reset(outer_trade_decision=outer_trade_decision, **kwargs)

    def generate_trade_decision(self, execute_result=None):
        if self.outer_trade_decision is not None:
            start_idx, _ = self.get_data_cal_avail_range(rtype='step')
            if not self.on_reset:
                for order in self.outer_trade_decision.order_list:
                    self.judge_cache[order.stock_id, order.direction] = self.predictor(order, start_idx)
            trade_decision = []
            trade_details = []
            trade_ensemble_cache = defaultdict(float)
            for order in self.outer_trade_decision.order_list:
                trade_details.append({
                    'instrument': order.stock_id,
                    'datetime': self.trade_calendar.get_step_time()[0],
                    'freq': self.trade_calendar.get_freq(),
                    'strategy': self.judge_cache[order.stock_id, order.direction]
                })
            trade_details = pd.DataFrame.from_records(trade_details)
            for idx, strategy in enumerate(self.strategies):
                sub_trade_decision = strategy.generate_trade_decision(execute_result=execute_result)
                for order in sub_trade_decision.order_list:
                    if isinstance(self.judge_cache[order.stock_id, order.direction], int):
                        if self.judge_cache[order.stock_id, order.direction] == idx:
                            trade_decision.append(order)
                    else:
                        trade_ensemble_cache[order.stock_id, order.direction] += \
                            self.judge_cache[order.stock_id, order.direction][idx] * order.amount
                if hasattr(sub_trade_decision, 'details'):
                    trade_details = trade_details.merge(
                        sub_trade_decision.details,
                        on=['instrument', 'datetime', 'freq']
                    )

            if trade_ensemble_cache:
                oh = self.trade_exchange.get_order_helper()
                for (stock_id, direction), amount in trade_ensemble_cache.items():
                    trade_decision.append(oh.create(stock_id, amount, direction))

            return TradeDecisionWithDetails(trade_decision, self, details=trade_details)

    def post_exe_step(self, execute_result=None):
        if self.outer_trade_decision is not None:
            for strategy in self.strategies:
                outer_calendar = self.outer_trade_decision.strategy.trade_calendar
                strategy = self.strategies[outer_calendar.get_trade_step()]
                if isinstance(strategy, RLStrategy):
                    strategy.post_exe_step(execute_result=execute_result)


class FirstOrLastStrategy(BaseStrategy):
    """
    To execute all the volumes at the very first step, or postpone it to the last.
    """

    def __init__(self, start: Literal['first', 'last'] = 'first'):
        self.start = start

    def reset(self, outer_trade_decision=None, **kwargs):
        super().reset(outer_trade_decision=outer_trade_decision, **kwargs)

        self.left = {}
        if outer_trade_decision is not None:
            for order in outer_trade_decision.order_list:
                self.left[order.stock_id, order.direction] = order.amount
            self.start_trade_step = 0 if self.start == 'first' else self.trade_calendar.get_trade_len() - 1

    def generate_trade_decision(self, execute_result=None):
        if self.outer_trade_decision is not None:
            if execute_result is not None:
                for e in execute_result:
                    self.left[e[0].stock_id, e[0].direction] -= e[0].deal_amount

            trade_decision = []
            for order in self.outer_trade_decision.order_list:
                oh = self.trade_exchange.get_order_helper()

                if self.trade_calendar.get_trade_step() >= self.start_trade_step \
                        and self.left[order.stock_id, order.direction] > 0:
                    trade_decision.append(oh.create(
                        order.stock_id,
                        min(order.amount, self.left[order.stock_id, order.direction]),
                        order.direction
                    ))

            return TradeDecisionWO(trade_decision, self)
        return TradeDecisionWO([], self)


class DecomposedStrategy(RLStrategy):
    def __init__(self):
        self.maintainer: Optional[StateMaintainer] = None

    @property
    def sample_state_pair(self) -> Tuple[IntraDaySingleAssetDataSchema, SAOEEpisodicState]:
        assert len(self.maintainer.samples) == len(self.maintainer.states) == 1
        return (
            list(self.maintainer.samples.values())[0],
            list(self.maintainer.states.values())[0],
        )

    def generate_trade_decision(self, execute_result=None):

        # get a decision from the outest loop
        exec_vol = yield self

        return TradeDecisionWO(
            self.maintainer.generate_orders(self.get_data_cal_avail_range(rtype='step'), [exec_vol]),
            self
        )


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


@SIMULATORS.register_module('intraday_sa_qlib')
class QlibSimulator(gym.Env):
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
