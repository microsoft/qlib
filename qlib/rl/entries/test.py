from __future__ import annotations

import pickle
from typing import Callable, Sequence

import pandas as pd
from tianshou.data import Collector
from tianshou.policy import BasePolicy

from qlib.rl.simulator import InitialStateType, Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.utils.data_queue import DataQueue
from qlib.rl.utils.env_wrapper import EnvWrapper
from qlib.rl.utils.finite_env import FiniteEnvType, finite_env_cls



INF = int(10 ** 18)


def backtest(
    simulator_fn: Callable[[InitialStateType], Simulator],
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    seed_set: Sequence[InitialStateType],
    policy: BasePolicy,
    reward: Reward | None = None,
    finite_env_type: FiniteEnvType = 'subproc'
):
    seed_iterator = DataQueue(seed_set)
    finite_env_cls(finite_env_type)

    with seed_iterator:
        env = EnvWrapper(simulator_fn, state_interpreter, action_interpreter, seed_iterator, reward)
        venv = env_factory(env_config, simulator_config, action, observation, reward, data_fn, logger)

        test_collector = Collector(policy, venv)
        policy.eval()
        print_log("All ready. Start backtest.", __name__)
        test_collector.collect(n_step=INF * len(venv))
    except StopIteration:
        pass
    finally:
        data_fn.cleanup()

    logger.write_summary()

    return pd.DataFrame.from_records(logger.logs), logger.history


def main(config):
    setup_experiment(config.runtime)

    dataset = config.data.source.build()
    action = config.action.build()
    observation = config.observation.build()
    reward = config.reward.build().eval()

    if config.runtime.debug:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(100)))
    data_fn = data_factory(config.data, dataset=dataset)
    logger = Logger(len(dataset))

    if config.network is not None:
        network = config.network.build()
        policy = config.policy.build(
            network=network, obs_space=observation.observation_space, action_space=action.action_space
        )
    else:
        policy = config.policy.build(obs_space=observation.observation_space, action_space=action.action_space)

    if use_cuda():
        policy.cuda()
    test_result, test_history = backtest(
        config.env, config.simulator, data_fn, logger, action, observation, reward, policy
    )
    test_result.to_csv(get_output_dir() / "metrics.csv", index=False)
    with (get_output_dir() / "history.pkl").open("wb") as f:
        pickle.dump(test_history, f)


if __name__ == "__main__":
    _config = RunConfig.fromcli()
    main(_config)
