# Introduction

This example tries to demonstrate how Qlib supports data without fixed shared frequency.

For example,
- Daily prices volume data are fixed-frequency data. The data comes in a fixed frequency (i.e. daily)
- Orders are not fixed data and they may come at any time point

To support such non-fixed-frequency, Qlib implements an Arctic-based backend.
Here is an example to import and query data based on this backend.

# Installation

Please refer to [the installation docs](https://docs.mongodb.com/manual/installation/) of mongodb.
Current version of script with default value tries to connect localhost **via default port without authentication**.

Run following command to install necessary libraries
```
pip install pytest coverage
pip install arctic  # NOTE: pip may fail to resolve the right package dependency !!! Please make sure the dependency are satisfied.
```

# Importing example data


1. (Optional) Please follow the first part of [this section](https://github.com/microsoft/qlib#data-preparation) to **get 1min data** of Qlib.
2. Please follow following steps to download example data
```bash
cd examples/orderbook_data/
python scripts/get_data.py qlib_data --target_dir "~/.qlib/orderbook_data" --name orderbook_data
```

3. Please import the example data to your mongo db
```bash
cd examples/orderbook_data/
python create_dataset.py initialize_library  # Initialization Libraries
python create_dataset.py import_data  # Initialization Libraries
```

# Query Examples

After importing these data, you run `example.py` to create some high-frequency features.
```bash
cd examples/orderbook_data/
pytest -s --disable-warnings example.py   # If you want run all examples
pytest -s --disable-warnings example.py::TestClass::test_exp_10  # If you want to run specific example
```


# Known limitations
Expression computing between different frequencies are not supported yet
