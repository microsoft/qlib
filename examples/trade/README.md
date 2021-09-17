# Universal Trading for Order Execution with Oracle Policy Distillation
This is the experiment code for our AAAI 2021 paper "[Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860)", including the implementations of all the compared methods in the paper and a general reinforcement learning framework for order execution in quantitative finance. 

## Abstract
As a fundamental problem in algorithmic trading, order execution aims at fulfilling a specific trading order, either liquidation or acquirement, for a given instrument. Towards effective execution strategy, recent years have witnessed the shift from the analytical view with model-based market assumptions to model-free perspective, i.e., reinforcement learning, due to its nature of sequential decision optimization. However, the noisy and yet imperfect market information that can be leveraged by the policy has made it quite challenging to build up sample efficient reinforcement learning methods to achieve effective order execution. In this paper, we propose a novel universal trading policy optimization framework to bridge the gap between the noisy yet imperfect market states and the optimal action sequences for order execution. Particularly, this framework leverages a policy distillation method that can better guide the learning of the common policy towards practically optimal execution by an oracle teacher with perfect information to approximate the optimal trading strategy. The extensive experiments have shown significant improvements of our method over various strong baselines, with reasonable trading actions.

## Environment Dependencies

### Dependencies

```
gym==0.17.3
torch==1.6.0
numba==0.51.2
numpy==1.19.1
pandas==1.1.3
tqdm==4.50.2
tianshou==0.3.0.post1
env==0.1.0
PyYAML==5.4.1
redis==3.5.3
```

### Environment Variable

`EXP_PATH` Absolute path to your config folder, we give folder `exp` as an example.

`OUTPUT_DIR` Absolute path to your log folder.

## Data Processing

For Feature processing, we take Yahoo dataset as an example, which can be precessed in `qlib/examples/highfreq/workflow.py` file. If you have a need to change your data storage path, you can change the `data_path` in `workflow.py`, and then do the following.

```
python workflow.py
```

For order generation, if you have changed change the the `data_path` in `workflow.py`, change `data_path` in `order_gen.py` again, then do the following.

```
python order_gen.py
```

## Training and backtest

### Config file

Config file is need to start our project, we take `PPO`, `OPDS` and `OPD` as an example in folder `exp/example`. If you want to use our given config, make sure the `data_path` you set before matches the config file. 

### Baseline method

To run a method, you can do the following.

```
python main.py --config={config_path}
```

Where `{config_path}` means the relative path from your config.yml to `EXP_PATH`.

If you need to run our given method such as PPO method, you can do the following.

```
python main.py --config=example/PPO/config.yml
```

### OPD method

OPD method is a multi step method, at first you should run OPDT as the teacher in OPD method.

```
python main.py --config=example/OPDT/config.yml
```

After training, find the `policy_best` file in your OPDT log file and copy it to `trade` file for backtest. Also you can change `policy_path` in the `example/OPDT_b/config.yml` to your `policy_best` file. Then run the backtest method.

```
python main.py --config=example/OPDT_b/config.yml
```

then processed feature from teacher. Remember to change `log_path` if you have changed `log_dir` in `OPDT_b/config.yml`.

```
python teacher_feature.py
```

and finally start our OPD method.

```
python main.py --config=example/OPD/config.yml
```

## Citation
You are more than welcome to citetmu our paper:
```
@inproceedings{fang2021universal,
  title={Universal Trading for Order Execution with Oracle Policy Distillation},
  author={Fang, Yuchen and Ren, Kan and Liu, Weiqing and Zhou, Dong and Zhang, Weinan and Bian, Jiang and Yu, Yong and Liu, Tie-Yan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```
