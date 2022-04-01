from typing import Protocol

from qlib.backtest.decision import BaseTradeDecision, Order, OrderHelper, TradeDecisionWO, TradeRangeByTime, TradeRange
from qlib.backtest.executor import BaseExecutor
from qlib.backtest.utils import CommonInfrastructure, TradeCalendarManager, get_start_end_idx
from qlib.strategy.base import BaseStrategy


class TradeDecisionWithDetails(TradeDecisionWO):
    def __init__(self, order_list: List[Order], strategy: BaseStrategy,
                 trade_range: Optional[Tuple[int, int]] = None, details: Optional[Any] = None):
        super().__init__(order_list, strategy, trade_range)
        self.details = details


class PostExectuionStrategy(Protocol):
    def post_exe_step(self, execute_result):
        """
        post process for each step of strategy this is design for RL Strategy,
        which require to update the policy state after each step

        NOTE: it is strongly coupled with RLNestedExecutor;
        """
        raise NotImplementedError("Please implement the `post_exe_step` method")


class RLStrategy(PostExectuionStrategy, BaseStrategy):
    """Qlib RL strategy that wraps neutrader components."""

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