=====================================================
Reinforcement Learning in Quantitative Trading
=====================================================

Reinforcement Learning
======================
Different from supervised learning tasks such as classification tasks and regression tasks. Another important paradigm in machine learning is Reinforcement Learning, 
which attempts to optimize an accumulative numerical reward signal by directly interacting with the environment under a few assumptions such as Markov Decision Process(MDP).

As demonstrated in the following figure, an RL system consists of four elements, 1)the agent 2) the environment the agent interacts with 3) the policy that the agent follows to take actions on the environment and 4)the reward signal from the environment to the agent. 
In general, the agent can perceive and interpret its environment, take actions and learn through reward, to seek long-term and maximum overall reward to achieve an optimal solution.

.. image:: ../../_static/img/RL_framework.png
   :width: 300
   :align: center 

RL attempts to learn to produce actions by trial and error. 
By sampling actions and then observing which one leads to our desired outcome, a policy is obtained to generate optimal actions. 
In contrast to supervised learning, RL learns this not from a label but from a time-delayed label called a reward. 
This scalar value lets us know whether the current outcome is good or bad. 
In a word, the target of RL is to take actions to maximize reward.

The Qlib Reinforcement Learning toolkit (QlibRL) is an RL platform for quantitative investment, which provides support to implement the RL algorithms in Qlib.


Potential Application Scenarios in Quantitative Trading
=======================================================
RL methods have already achieved outstanding achievement in many applications, such as game playing, resource allocating, recommendation, marketing and advertising, etc.
Investment is always a continuous process, taking the stock market as an example, investors need to control their positions and stock holdings by one or more buying and selling behaviors, to maximize the investment returns.
Besides, each buy and sell decision is made by investors after fully considering the overall market information and stock information. 
From the view of an investor, the process could be described as a continuous decision-making process generated according to interaction with the market, such problems could be solved by the RL algorithms. 
Following are some scenarios where RL can potentially be used in quantitative investment.

Portfolio Construction
----------------------
Portfolio construction is a process of selecting securities optimally by taking a minimum risk to achieve maximum returns. With an RL-based solution, an agent allocates stocks at every time step by obtaining information for each stock and the market. The key is to develop of policy for building a portfolio and make the policy able to pick the optimal portfolio. 

Order Execution
---------------
As a fundamental problem in algorithmic trading, order execution aims at fulfilling a specific trading order, either liquidation or acquirement, for a given instrument. Essentially, the goal of order execution is twofold: it not only requires to fulfill the whole order but also targets a more economical execution with maximizing profit gain (or minimizing capital loss). The order execution with only one order of liquidation or acquirement is called single-asset order execution.

Considering stock investment always aim to pursue long-term maximized profits, it usually manifests as a sequential process of continuously adjusting the asset portfolios, execution for multiple orders, including order of liquidation and acquirement, brings more constraints and makes the sequence of execution for different orders should be considered, e.g. before executing an order to buy some stocks, we have to sell at least one stock. The order execution with multiple assets is called multi-asset order execution. 

According to the order execution’s trait of sequential decision-making, an RL-based solution could be applied to solve the order execution. With an RL-based solution, an agent optimizes execution strategy by interacting with the market environment. 

With QlibRL, the RL algorithm in the above scenarios can be easily implemented.

Nested Portfolio Construction and Order Executor
------------------------------------------------
QlibRL makes it possible to jointly optimize different levels of strategies/models/agents. Take `Nested Decision Execution Framework <https://github.com/microsoft/qlib/blob/main/examples/nested_decision_execution>`_ as an example, the optimization of order execution strategy and portfolio management strategies can interact with each other to maximize returns.
