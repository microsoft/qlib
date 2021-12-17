# Introduction
This folder contains 2 examples
- A high-frequency dataset example
- An example of predicting the price trend in high-frequency data

## High-Frequency Dataset

This dataset is an example for RL high frequency trading.

### Get High-Frequency Data

Get high-frequency data by running the following command:
```bash
    python workflow.py get_data
```

### Dump & Reload & Reinitialize the Dataset


The High-Frequency Dataset is implemented as `qlib.data.dataset.DatasetH` in the `workflow.py`. `DatatsetH` is the subclass of [`qlib.utils.serial.Serializable`](https://qlib.readthedocs.io/en/latest/advanced/serial.html), whose state can be dumped in or loaded from disk in `pickle` format.

### About Reinitialization

After reloading `Dataset` from disk, `Qlib` also support reinitializing the dataset. It means that users can reset some states of `Dataset` or `DataHandler` such as `instruments`, `start_time`, `end_time` and `segments`, etc.,  and generate new data according to the states.

The example is given in `workflow.py`, users can run the code as follows.

### Run the Code

Run the example by running the following command:
```bash
    python workflow.py dump_and_load_dataset
```

## Benchmarks Performance (predicting the price trend in high-frequency data)

Here are the results of models for predicting the price trend in high-frequency data. We will keep updating benchmark models in future.

| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Long precision| Short Precision | Long-Short Average Return | Long-Short Average Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | Alpha158 | 0.0349±0.00 | 0.3805±0.00| 0.0435±0.00 | 0.4724±0.00 | 0.5111±0.00 | 0.5428±0.00 | 0.000074±0.00 | 0.2677±0.00 |
