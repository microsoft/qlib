This folder contains a simple example of how to run Qlib RL. It contains:

```
.
├── experiment_config
│   ├── backtest       # Backtest config
│   └── training       # Training config
├── README.md          # Readme (the current file)
└── scripts            # Scripts for data pre-processing
```

## Data preparation

Use [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) to download data:

```
azcopy copy https://qlibpublic.blob.core.windows.net/data/rl/qlib_rl_example_data ./ --recursive
mv qlib_rl_example_data data
```

The downloaded data will be placed at `./data`. The original data are in `data/csv`. To create all data needed by the case, run:

```
bash scripts/data_pipeline.sh
```

After the execution finishes, the `data/` directory should be like:

```
data
├── backtest_orders.csv
├── bin
├── csv
├── pickle
├── pickle_dataframe
└── training_order_split
```

## Run training

Run:

```
python -m qlib.rl.contrib.train_onpolicy --config_path ./experiment_config/training/config.yml
```

After training, checkpoints will be stored under `checkpoints/`.

## Run backtest

```
python -m qlib.rl.contrib.backtest --config_path ./experiment_config/backtest/config.yml
```

The backtest workflow will use the trained model in `checkpoints/`. The backtest summary can be found in `outputs/`.
