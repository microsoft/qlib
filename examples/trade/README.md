# Universal Trading for Order Execution with Oracle Policy Distillation
This is the experiment code for our AAAI 2021 paper "[Universal Trading for Order Execution with Oracle Policy Distillation](https://seqml.github.io/opd/opd_aaai21.pdf)", including the implementations of all the compared methods in the paper and a general reinforcement learning framework for order execution in quantitative finance. 

### Abstract
As a fundamental problem in algorithmic trading, order execution aims at fulfilling a specific trading order, either liquidation or acquirement, for a given instrument. Towards effective execution strategy, recent years have witnessed the shift from the analytical view with model-based market assumptions to model-free perspective, i.e., reinforcement learning, due to its nature of sequential decision optimization. However, the noisy and yet imperfect market information that can be leveraged by the policy has made it quite challenging to build up sample efficient reinforcement learning methods to achieve effective order execution. In this paper, we propose a novel universal trading policy optimization framework to bridge the gap between the noisy yet imperfect market states and the optimal action sequences for order execution. Particularly, this framework leverages a policy distillation method that can better guide the learning of the common policy towards practically optimal execution by an oracle teacher with perfect information to approximate the optimal trading strategy. The extensive experiments have shown significant improvements of our method over various strong baselines, with reasonable trading actions.

### Instruction
We will update the data processing scripts and the experiment workflow soon.

### Citation
You are more than welcome to cite our paper:
```
@inproceedings{fang2021universal,
  title={Universal Trading for Order Execution with Oracle Policy Distillation},
  author={Fang, Yuchen and Ren, Kan and Liu, Weiqing and Zhou, Dong and Zhang, Weinan and Bian, Jiang and Yu, Yong and Liu, Tie-Yan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```
