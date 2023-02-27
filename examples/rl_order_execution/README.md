# RL Example for Order Execution

This folder comprises an example of Reinforcement Learning (RL) workflows for order execution scenario, including both training workflows and backtest workflows.

## Data Processing

### Get Data

```
python ../../scripts/get_data.py qlib_data --target_dir ./data/bin --region hs300 --interval 5min
```

### Generate Pickle-Style Data

To run codes in this example, we need data in pickle format. To achieve this, run following commands (might need a few minutes to finish):

```
python scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml;
python scripts/collect_pickle_dataframe.py;
python scripts/gen_training_orders.py;
python scripts/merge_orders.py;
```

When finished, the structure under `data/` should be:

```
data
├── bin
├── orders
├── pickle
└── pickle_dataframe
```

## Training

Each training task is specified by a config file. The config file for task `TASKNAME` is `exp_configs/train_TASKNAME.yml`. This example provides two training tasks:

- **PPO**: Method proposed by IJCAL 2020 paper "[An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization](https://www.ijcai.org/proceedings/2020/0627.pdf)".
- **OPDS**: Method proposed by AAAI 2021 paper "[Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860)".

The main differece between these two methods is their reward functions. Please see their config files for details.

Take OPDS as an example, to run the training workflow, run:

```
python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_opds.yml --run_backtest
```

Metrics, logs, and checkpoints will be stored under `outputs/opds` (configured by `exp_configs/train_opds.yml`). 

## Backtest

Once the training workflow has completed, the trained model can be used for the backtesting workflow. Still take OPDS as an example, once training is finished, the latest checkpoint of the model can be found at `outputs/opds/checkpoints/latest.pth`. To run backtest workflow:

1. Uncomment the `weight_file` parameter in `exp_configs/train_opds.yml` (it is commented by default). While it is possible to run the backtesting workflow without setting a checkpoint, this will lead to randomly initialized model results, thus making them meaningless.
2. Run `python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_opds.yml`.

The backtest result is stored in `outputs/checkpoints/backtest_result.csv`.

In addition to OPDS and PPO, we also provide TWAP ([Time-weighted average price](https://en.wikipedia.org/wiki/Time-weighted_average_price)) as a weak baseline. The config file for TWAP is `exp_configs/backtest_twap.yml`.

## Benchmarks

To accurately evaluate the performance of models using Reinforcement Learning algorithms, it's best to run experiments multiple times and compute the average performance across all trials. However, given the time-consuming nature of model training, this is not always feasible. An alternative approach is to run each training task only once, selecting the 10 checkpoints with the highest validation performance to simulate multiple trials. In this example, we use "Price Advantage (PA)" as the metric for selecting these checkpoints. The average performance of these 10 checkpoints on the testing set is as follows:

| **Model** | **PA mean with std.** |
|-----------|-----------------------|
| OPDS      |  0.4785 ± 0.7815      |
| PPO       | -1.0935 ± 0.0922      |
| TWAP      |  ≈ 0.0 ± 0.0          |

The table above also includes TWAP as a rule-based baseline. The ideal PA of TWAP should be 0.0, however, in this example, the order execution is divided into two steps: first, the order is split equally among each half hour, and then each five minutes within each half hour. Since trading is forbidden during the last five minutes of the day, this approach may slightly differ from traditional TWAP over the course of a full day (as there are 5 minutes missing in the last "half hour"). Therefore, the PA of TWAP can be considered as a number that is close to 0.0. To verify this, you may run a TWAP backtest and check the results.
