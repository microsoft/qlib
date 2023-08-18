
========
Guidance
========
.. currentmodule:: qlib

QlibRL can help users quickly get started and conveniently implement quantitative strategies based on reinforcement learning(RL) algorithms. For different user groups, we recommend the following guidance to use QlibRL.

Beginners to Reinforcement Learning Algorithms
==============================================
Whether you are a quantitative researcher who wants to understand what RL can do in trading or a learner who wants to get started with RL algorithms in trading scenarios, if you have limited knowledge of RL and want to shield various detailed settings to quickly get started with RL algorithms, we recommend the following sequence to learn qlibrl:
 - Learn the fundamentals of RL in `part1 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#reinforcement-learning>`_.
 - Understand the trading scenarios where RL methods can be applied in `part2 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#potential-application-scenarios-in-quantitative-trading>`_.
 - Run the examples in `part3 <https://qlib.readthedocs.io/en/latest/component/rl/quickstart.html>`_ to solve trading problems using RL.
 - If you want to further explore QlibRL and make some customizations, you need to first understand the framework of QlibRL in `part4 <https://qlib.readthedocs.io/en/latest/component/rl/framework.html>`_ and rewrite specific components according to your needs.

Reinforcement Learning Algorithm Researcher
==============================================
If you are already familiar with existing RL algorithms and dedicated to researching RL algorithms but lack domain knowledge in the financial field, and you want to validate the effectiveness of your algorithms in financial trading scenarios, we recommend the following steps to get started with QlibRL:
 - Understand the trading scenarios where RL methods can be applied in `part2 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#potential-application-scenarios-in-quantitative-trading>`_.
 - Choose an RL application scenario (currently, QlibRL has implemented two scenario examples: order execution and algorithmic trading). Run the example in `part3 <https://qlib.readthedocs.io/en/latest/component/rl/quickstart.html>`_ to get it working.
 - Modify the `policy <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/policy.py>`_ part to incorporate your own RL algorithm.

Quantitative Researcher
=======================
If you have a certain level of financial domain knowledge and coding skills, and you want to explore the application of RL algorithms in the investment field, we recommend the following steps to explore QlibRL:
 - Learn the fundamentals of RL in `part1 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#reinforcement-learning>`_.
 - Understand the trading scenarios where RL methods can be applied in `part2 <https://qlib.readthedocs.io/en/latest/component/rl/overall.html#potential-application-scenarios-in-quantitative-trading>`_.
 - Run the examples in `part3 <https://qlib.readthedocs.io/en/latest/component/rl/quickstart.html>`_ to solve trading problems using RL.
 - Understand the framework of QlibRL in `part4 <https://qlib.readthedocs.io/en/latest/component/rl/framework.html>`_.
 - Choose a suitable RL algorithm based on the characteristics of the problem you want to solve (currently, QlibRL supports PPO and DQN algorithms based on tianshou).
 - Design the MDP (Markov Decision Process) process based on market trading rules and the problem you want to solve. Refer to the example in order execution and make corresponding modifications to the following modules: `State <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/state.py#L70>`_, `Metrics <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/state.py#L18>`_, `ActionInterpreter <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/interpreter.py#L199>`_, `StateInterpreter <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/interpreter.py#L68>`_, `Reward <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/reward.py>`_, `Observation <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/interpreter.py#L44>`_, `Simulator <https://github.com/microsoft/qlib/blob/main/qlib/rl/order_execution/simulator_simple.py>`_.