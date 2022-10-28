.. _rl:
========================================================================
Reinforcement Learning in Quantitative Trading
========================================================================
.. currentmodule:: qlib

Reinforcement Learning
============

Reinforcement learning attempts to optimize an accumulative numerical reward signal by directly interacting with the environment under a few assumptions such as Markov Decision Process(MDP), which is the third basic machine learning method besides supervised and unsupervised learning.

As demonstrated in the following figure, an RL system consists of four elements, 1)the agent 2) the environment the agent interacts with 3) the policy that the agent follows to take actions on the environment and 4)the reward signal from the environment to the agent. 
In general, the agent can perceive and interpret its environment, take actions and learn through reward, to seek long-term and maximum overall reward to achieve an optimal solution.

.. image:: ../_static/img/RL_framework.png
   :width: 300
   :align: center

The Qlib Reinforcement Learning toolkit (QlibRL) is an RL platform for quantitative investment, which provides support to implement RL algorithms in Qlib.


Potential Application Scenarios in Quantitative Trading
============
RL methods have already achieved outstanding achievement in many applications, such as game playing, resource allocating, recommendation, marketing and advertising, etc.
In quantitative investment, RL potentially can be used in many scenarios as well.

Portfolio Construction
------------
Portfolio construction is a process of selecting securities optimally by taking a minimum risk to achieve maximum returns. With an RL-based solution, an agent allocates stocks at every time step by obtaining information for each stock and the market. The key is to develop of policy for building a portfolio and make the policy able to pick the optimal portfolio. 

Order Execution
------------
As a fundamental problem in algorithmic trading, order execution aims at fulfilling a specific trading order, either liquidation or acquirement, for a given instrument. Essentially, the goal of order execution is twofold: it not only requires to fulfill the whole order but also targets a more economical execution with maximizing profit gain (or minimizing capital loss). The order execution with only one order of liquidation or acquirement is called single-asset order execution.

Considering stock investment always aim to pursue long-term maximized profits, it usually manifests as a sequential process of continuously adjusting the asset portfolios, execution for multiple orders, including order of liquidation and acquirement, brings more constraints and makes the sequence of execution for different orders should be considered, e.g. before executing an order to buy some stocks, we have to sell at least one stock. The order execution with multiple assets is called multi-asset order execution. 

According to the order execution’s trait of sequential decision-making, an RL-based solution could be applied to solve the order execution. With an RL-based solution, an agent optimizes execution strategy by interacting with the market environment. 

With ``QlibRL``, the RL algorithm in the above scenarios can be easily implemented.

Nested Portfolio Construction and Order Executor
------------
``QlibRL`` makes it possible to jointly optimize different levels of strategies/models/agents. Take `Nested Decision Execution Framework <https://github.com/microsoft/qlib/blob/main/examples/nested_decision_execution>`_ as an example, the optimization of order execution strategy and portfolio management strategies can interact with each other to maximize returns.


Quick Start
============
``QlibRL`` provides an example of an implementation of a single asset order execution task and the following is an example of the config file to train with QlibRL.

.. code-block:: text

    simulator:
        time_per_step: 30
        vol_limit: null
    env:
        concurrency: 1
        parallel_mode: dummy
    action_interpreter:
        class: CategoricalActionInterpreter
        kwargs:
            values: 14
            max_step: 8
        module_path: qlib.rl.order_execution.interpreter
    state_interpreter:
        class: FullHistoryStateInterpreter
        kwargs:
            data_dim: 6
            data_ticks: 240
            max_step: 8
            processed_data_provider:
                class: PickleProcessedDataProvider
                module_path: qlib.rl.data.pickle_styled
                kwargs:
                    data_dir: ./data/pickle_dataframe/feature
        module_path: qlib.rl.order_execution.interpreter
    reward:
        class: PAPenaltyReward
        kwargs:
            penalty: 100.0
        module_path: qlib.rl.order_execution.reward
    data:
        source:
            order_dir: ./data/training_order_split
            data_dir: ./data/pickle_dataframe/backtest
            total_time: 240
            default_start_time: 0
            default_end_time: 240
            proc_data_dim: 6
        num_workers: 0
        queue_size: 20
    network:
        class: Recurrent
        module_path: qlib.rl.order_execution.network
    policy:
        class: PPO
        kwargs:
            lr: 0.0001
        module_path: qlib.rl.order_execution.policy
    runtime:
        seed: 42
        use_cuda: false
    trainer:
        max_epoch: 2
        repeat_per_collect: 5
        earlystop_patience: 2
        episode_per_collect: 20
        batch_size: 16
        val_every_n_epoch: 1
        checkpoint_path: ./checkpoints
        checkpoint_every_n_iters: 1


And the config file for backtesting:

.. code-block:: text

    order_file: ./data/backtest_orders.csv
    start_time: "9:45"
    end_time: "14:44"
    qlib:
    provider_uri_1min: ./data/bin
    feature_root_dir: ./data/pickle
    feature_columns_today: [
        "$open", "$high", "$low", "$close", "$vwap", "$volume",
    ]
    feature_columns_yesterday: [
        "$open_v1", "$high_v1", "$low_v1", "$close_v1", "$vwap_v1", "$volume_v1",
    ]
    exchange:
    limit_threshold: ['$close == 0', '$close == 0']
    deal_price: ["If($close == 0, $vwap, $close)", "If($close == 0, $vwap, $close)"]
    volume_threshold:
        all: ["cum", "0.2 * DayCumsum($volume, '9:45', '14:44')"]
        buy: ["current", "$close"]
        sell: ["current", "$close"]
    strategies: 
    30min: 
        class: TWAPStrategy
        module_path: qlib.contrib.strategy.rule_strategy
        kwargs: {}
    1day: 
        class: SAOEIntStrategy
        module_path: qlib.rl.order_execution.strategy
        kwargs:
        state_interpreter:
            class: FullHistoryStateInterpreter
            module_path: qlib.rl.order_execution.interpreter
            kwargs:
            max_step: 8
            data_ticks: 240
            data_dim: 6
            processed_data_provider:
                class: PickleProcessedDataProvider
                module_path: qlib.rl.data.pickle_styled
                kwargs:
                data_dir: ./data/pickle_dataframe/feature
        action_interpreter: 
            class: CategoricalActionInterpreter
            module_path: qlib.rl.order_execution.interpreter
            kwargs: 
            values: 14
            max_step: 8
        network: 
            class: Recurrent
            module_path: qlib.rl.order_execution.network
            kwargs: {}
        policy: 
            class: PPO
            module_path: qlib.rl.order_execution.policy
            kwargs: 
                lr: 1.0e-4
                weight_file: ./checkpoints/latest.pth
    concurrency: 5

With the above config files, you can start training the agent by the following command:

.. code-block:: console

    $ python qlib/rl/contrib/train_onpolicy.py --config_path train_config.yml

After the training, you can backtest with the following command:

.. code-block:: console

    $ python qlib/rl/contrib/backtest.py --config_path backtest_config.yml

In that case, `qlib.rl.order_execution.simulator_qlib.SingleAssetOrderExecution <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/simulator_qlib.py>`_ and `qlib.rl.order_execution.simulator_simple.SingleAssetOrderExecutionSimple <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/simulator_simple.py>`_ as examples for simulator, `StateInterpreter <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/interpreter.py>`_ and `ActionInterpreter <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/interpreter.py>`_ as examples for interpreter, and `qlib.rl.order_execution.reward.PAPenaltyReward <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/reward.py>`_ as an example for reward.
For the single asset order execution task, if developers have already defined their simulator/interpreters/reward function/policy, they could launch the training and backtest pipeline by simply modifying the corresponding settings in the config files.
The details about the example can be found `here <../../examples/rl/README.md>`_. 

In the future, we will provide more examples for different scenarios such as RL-based portfolio construction.

The Framework of QlibRL
============
QlibRL contains a full set of components that cover the entire lifecycle of an RL pipeline, including building the simulator of the market, shaping states & actions, training policies (strategies), and backtesting strategies in the simulated environment.

QlibRL is basically implemented with the support of Tianshou and Gym frameworks. The high-level structure of QlibRL is demonstrated below:

.. image:: ../_static/img/QlibRL_framework.png
   :width: 600
   :align: center

Here, we briefly introduce each component in the figure.

EnvWrapper
------------
EnvWrapper is the complete capsulation of the simulated environment. It receives actions from outside (policy/strategy/agent), simulates the changes in the market, and then replies rewards and updated states, thus forming an interaction loop.

In QlibRL, EnvWrapper is a subclass of gym.Env, so it implements all necessary interfaces of gym.Env. Any classes or pipelines that accept gym.Env should also accept EnvWrapper. Developers do not need to implement their own EnvWrapper to build their own environment. Instead, they only need to implement 4 components of the EnvWrapper:

- `Simulator`
    The simulator is the core component responsible for the environment simulation. Developers could implement all the logic that is directly related to the environment simulation in the Simulator in any way they like. In QlibRL, there are already two implementations of Simulator for single asset trading: 1) ``SingleAssetOrderExecution``, which is built based on Qlib's backtest toolkits and hence considers a lot of practical trading details but is slow. 2) ``SimpleSingleAssetOrderExecution``, which is built based on a simplified trading simulator, which ignores a lot of details (e.g. trading limitations, rounding) but is quite fast.
- `State interpreter` 
    The state interpreter is responsible for "interpret" states in the original format (format provided by the simulator) into states in a format that the policy could understand. For example, transform unstructured raw features into numerical tensors.
- `Action interpreter` 
    The action interpreter is similar to the state interpreter. But instead of states, it interprets actions generated by the policy, from the format provided by the policy to the format that is acceptable to the simulator.
- `Reward function` 
    The reward function returns a numerical reward to the policy after each time the policy takes an action. 

EnvWrapper will organically organize these components. Such decomposition allows for better flexibility in development. For example, if the developers want to train multiple types of policies in the same environment, they only need to design one simulator and design different state interpreters/action interpreters/reward functions for different types of policies.

QlibRL has well-defined base classes for all these 4 components. All the developers need to do is define their own components by inheriting the base classes and then implementing all interfaces required by the base classes.

Policy
------------
QlibRL directly uses Tianshou's policy. Developers could use policies provided by Tianshou off the shelf, or implement their own policies by inheriting Tianshou's policies.

Training Vessel & Trainer
------------
As stated by their names, training vessels and trainers are helper classes used in training. A training vessel is a ship that contains a simulator/interpreters/reward function/policy, and it controls algorithm-related parts of training. Correspondingly, the trainer is responsible for controlling the runtime parts of training.

As you may have noticed, a training vessel itself holds all the required components to build an EnvWrapper rather than holding an instance of EnvWrapper directly. This allows the training vessel to create duplicates of EnvWrapper dynamically when necessary (for example, under parallel training).

With a training vessel, the trainer could finally launch the training pipeline by simple, Scikit-learn-like interfaces (i.e., `trainer.fit()`).


Base Class & Interface 
============
``Qlib`` provides a set of APIs for developers to further simplify their development such as base classes for Interpreter, Simulator, Reward.

.. automodule:: qlib.rl
   :members: