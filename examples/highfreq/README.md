# High-Frequency Dataset

This dataset is an example for RL high frequency trading.

## Get High-Frequency Data

Get high-frequency data by running the following command:
```bash
    python workflow.py get_data
```

## Dump & Reload & Reinitialize the Dataset


The High-Frequency Dataset is implemented as `qlib.data.dataset.DatasetH` in the `workflow.py`. `DatatsetH` is the subclass of `qlib.utils.serial.Serializable`, which supports being dumped in or loaded from disk in `pickle` format.

### About Reinitialization

After reloading `Dataset` from disk, `Qlib` also support reinitialize the dataset. It means that users can reset some config of `Dataset` or `DataHandler` such as `instruments`, `start_time`, `end_time` and `segmens`, etc.

The example is given in `workflow.py`, users can run the code as follows.

### Run the Code

Run the example by running the following command:
```bash
    python workflow.py dump_and_load_dataset
```