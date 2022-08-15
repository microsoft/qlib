.. _online:

Online
======
.. currentmodule:: qlib

Introduction
------------

Welcome to use Online, this module simulates what will be like if we do the real trading use our model and strategy.

Just like Estimator and other modules in Qlib, you need to determine parameters through the configuration file,
and in this module, you need to add an account in a folder to do the simulation. Then in each coming day,
this module will use the newest information to do the trade for your account,
the performance can be viewed at any time using the API we defined.

Each account will experience the following processes, the ‘pred_date’ represents the date you predict the target
positions after trading, also, the ‘trade_date’ is the date you do the trading.

- Generate the order list (pre_date)
- Execute the order list (trade_date)
- Update account (trade_date)

In the meantime, you can just create an account and use this module to test its performance in a period.

- Simulate (start_date, end_date)

This module need to save your account in a folder, the model and strategy will be saved as pickle files,
and the position and report will be saved as excel.
The file structure can be viewed at fileStruct_.


Example
-------

Let's take an example,

.. note:: Make sure you have the latest version of `qlib` installed.

If you want to use the models and data provided by `qlib`, you only need to do as follows.

Firstly, write a simple configuration file as following,

.. code-block:: YAML

    strategy:
        class: TopkAmountStrategy
        module_path: qlib.contrib.strategy
        args:
            market: csi500
            trade_freq: 5

    model:
        class: ScoreFileModel
        module_path: qlib.contrib.online.online_model
        args:
            loss: mse
            model_path: ./model.bin

    init_cash: 1000000000

We then can use this command to create a folder and do trading from 2017-01-01 to 2018-08-01.

.. code-block:: bash

    online simulate -id v-test -config ./config/config.yaml -exchange_config ./config/exchange.yaml -start 2017-01-01 -end 2018-08-01 -path ./user_data/

The start date (2017-01-01) is the add date of the user, which also is the first predict date,
and the end date (2018-08-01) is the last trade date. You can use "`online generate -date 2018-08-02...`"
command to continue generate the order_list at next trading date.

If Your account was saved in "./user_data/", you can see the performance of your account compared to a benchmark by

.. code-block:: bash

    >> online show -id v-test -path ./user_data/ -bench SH000905

    ...
    Result of porfolio:
                                                      risk
    excess_return_without_cost mean               0.000605
                               std                0.005481
                               annualized_return  0.152373
                               information_ratio  1.751319
                               max_drawdown      -0.059055
    excess_return_with_cost    mean               0.000410
                               std                0.005478
                               annualized_return  0.103265
                               information_ratio  1.187411
                               max_drawdown      -0.075024


Here 'SH000905' represents csi500 and 'SH000300' represents csi300

Manage your account
-------------------

Any account processed by `online` should be saved in a folder. you can use commands
defined to manage your accounts.

- add an new account
    This will add an new account with user_id='v-test', add_date='2019-10-15' in ./user_data.

    .. code-block:: bash

        >> online add_user -id {user_id} -config {config_file} -path {folder_path} -date {add_date}
        >> online add_user -id v-test -config config.yaml -path ./user_data/ -date 2019-10-15

- remove an account
    .. code-block:: bash

        >> online remove_user -id {user_id} -path {folder_path}
        >> online remove_user -id v-test -path ./user_data/

- show the performance
    Here benchmark indicates the baseline is to be compared with yours.

    .. code-block:: bash

        >> online show -id {user_id} -path {folder_path} -bench {benchmark}
        >> online show -id v-test -path ./user_data/ -bench SH000905

The default value of all the parameter 'date' below is trade date
(will be today if today is trading date and information has been updated in `qlib`).

The 'generate' and 'update' will check whether input date is valid, the following 3 processes should
be called at each trading date.

- generate the order list
    generate the order list at trade date, and save them in {folder_path}/{user_id}/temp/ as a json file.

    .. code-block:: bash

        >> online generate -date {date} -path {folder_path}
        >> online generate -date 2019-10-16 -path ./user_data/

- execute the order list
    execute the order list and generate the transactions result in {folder_path}/{user_id}/temp/ at trade date

    .. code-block:: bash

        >> online execute -date {date} -exchange_config {exchange_config_path} -path {folder_path}
        >> online execute -date 2019-10-16 -exchange_config ./config/exchange.yaml -path ./user_data/

    A simple exchange config file can be as

    .. code-block:: yaml

        open_cost: 0.003
        close_cost: 0.003
        limit_threshold: 0.095
        deal_price: vwap


- update accounts
    update accounts in "{folder_path}/" at trade date

    .. code-block:: bash

        >> online update -date {date} -path {folder_path}
        >> online update -date 2019-10-16 -path ./user_data/

API
---

All those operations are based on defined in `qlib.contrib.online.operator`

.. automodule:: qlib.contrib.online.operator

.. _fileStruct:

File structure
--------------

'user_data' indicates the root of folder.
Name that bold indicates it’s a folder, otherwise it’s a document.

.. code-block:: yaml

    {user_folder}
    │   users.csv: (Init date for each users)
    │
    └───{user_id1}: (users' sub-folder to save their data)
    │   │   position.xlsx
    │   │   report.csv
    │   │   model_{user_id1}.pickle
    │   │   strategy_{user_id1}.pickle
    │   │
    │   └───score
    │   │   └───{YYYY}
    │   │       └───{MM}
    │   │           │   score_{YYYY-MM-DD}.csv
    │   │
    │   └───trade
    │       └───{YYYY}
    │           └───{MM}
    │               │   orderlist_{YYYY-MM-DD}.json
    │               │   transaction_{YYYY-MM-DD}.csv
    │
    └───{user_id2}
    │   │   position.xlsx
    │   │   report.csv
    │   │   model_{user_id2}.pickle
    │   │   strategy_{user_id2}.pickle
    │   │
    │   └───score
    │   └───trade
    ....


Configuration file
------------------

The configure file used in `online` should contain the model and strategy information.

About the model
~~~~~~~~~~~~~~~

First, your configuration file needs to have a field about the model,
this field and its contents determine the model we used when generating score at predict date.

Followings are two examples for ScoreFileModel and a model that read a score file and return score at trade date.

.. code-block:: YAML

     model:
        class: ScoreFileModel
        module_path: qlib.contrib.online.OnlineModel
        args:
            loss: mse

.. code-block:: YAML

     model:
        class: ScoreFileModel
        module_path: qlib.contrib.online.OnlineModel
        args:
            score_path: <your score path>

If your model doesn't belong to above models, you need to coding your model manually.
Your model should be a subclass of models defined in 'qlib.contfib.model'. And it must
contains 2 methods used in `online` module.


About the strategy
~~~~~~~~~~~~~~~~~~

Your need define the strategy used to generate the order list at predict date.

Followings are two examples for a TopkAmountStrategy

.. code-block:: YAML

    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy.strategy
        args:
            topk: 100
            n_drop: 10

Generated files
---------------

The 'online_generate' command will create the order list at {folder_path}/{user_id}/temp/,
the name of that is orderlist_{YYYY-MM-DD}.json, YYYY-MM-DD is the date that those orders to be executed.

The format of json file is like

.. code-block:: python

    {
        'sell': {
                {'$stock_id1': '$amount1'},
                {'$stock_id2': '$amount2'}, ...
                },
        'buy': {
                {'$stock_id1': '$amount1'},
                {'$stock_id2': '$amount2'}, ...
                }
    }

Then after executing the order list (either by 'online_execute' or other executors), a transaction file
will be created also at {folder_path}/{user_id}/temp/.
