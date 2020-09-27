Changelog
====================
Here you can see the full list of changes between each QLib release.

Version 0.1.0
--------------------
This is the initial release of QLib library.

Version 0.1.1
--------------------
Performance optimize. Add more features and operators.

Version 0.1.2
--------------------
- Support operator syntax. Now ``High() - Low()`` is equivalent to ``Sub(High(), Low())``.   
- Add more technical indicators.

Version 0.1.3
--------------------
Bug fix and add instruments filtering mechanism.

Version 0.2.0
--------------------
- Redesign ``LocalProvider`` database format for performance improvement.
- Support load features as string fields.
- Add scripts for database construction.
- More operators and technical indicators.

Version 0.2.1
--------------------
- Support registering user-defined ``Provider``.
- Support use operators in string format, e.g. ``['Ref($close, 1)']`` is valid field format.
- Support dynamic fields in ``$some_field`` format. And exising fields like ``Close()`` may be deprecated in the future.

Version 0.2.2
--------------------
- Add ``disk_cache`` for reusing features (enabled by default).
- Add ``qlib.contrib`` for experimental model construction and evaluation.


Version 0.2.3
--------------------
- Add ``backtest`` module
- Decoupling the Strategy, Account, Position, Exchange from the backtest module

Version 0.2.4
--------------------
- Add ``profit attribution`` module
- Add ``rick_control`` and ``cost_control`` strategies
  
Version 0.3.0
--------------------
- Add ``estimator`` module

Version 0.3.1
--------------------
- Add ``filter`` module

Version 0.3.2
--------------------
- Add real price trading, if the ``factor`` field in the data set is incomplete, use ``adj_price`` trading
- Refactor ``handler`` ``launcher`` ``trainer`` code
- Support ``backtest`` configuration parameters in the configuration file
- Fix bug in position ``amount`` is 0
- Fix bug of ``filter`` module

Version 0.3.3
-------------------
- Fix bug of ``filter`` module

Version 0.3.4
--------------------
- Support for ``finetune model``
- Refactor ``fetcher`` code

Version 0.3.5
--------------------
- Support multi-label training, you can provide multiple label in ``handler``. (But LightGBM doesn't support due to the algorithm itself)
- Refactor ``handler`` code, dataset.py is no longer used, and you can deploy your own labels and features in ``feature_label_config``
- Handler only offer DataFrame. Also, ``trainer`` and model.py only receive DataFrame
- Change ``split_rolling_data``, we roll the data on market calender now, not on normal date
- Move some date config from ``handler`` to ``trainer``

Version 0.4.0
--------------------
- Add `data` package that holds all data-related codes
- Reform the data provider structure
- Create a server for data centralized management `qlib-server<https://amc-msra.visualstudio.com/trading-algo/_git/qlib-server>`_
- Add a `ClientProvider` to work with server
- Add a pluggable cache mechanism
- Add a recursive backtracking algorithm to inspect the furthest reference date for an expression

.. note::
    The ``D.instruments`` function does not support ``start_time``, ``end_time``, and ``as_list`` parameters, if you want to get the results of previous versions of ``D.instruments``, you can do this:


    >>> from qlib.data import D
    >>> instruments = D.instruments(market='csi500')
    >>> D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)


Version 0.4.1
--------------------
- Add support Windows
- Fix ``instruments`` type bug
- Fix ``features`` is empty bug(It will cause failure in updating)
- Fix ``cache`` lock and update bug
- Fix use the same cache for the same field (the original space will add a new cache)
- Change "logger handler" from config
- Change model load support 0.4.0 later
- The default value of the ``method`` parameter of ``risk_analysis`` function is changed from **ci** to **si**


Version 0.4.2
--------------------
- Refactor DataHandler
- Add ``ALPHA360`` DataHandler


Version 0.4.3
--------------------
- Implementing Online Inference and Trading Framework
- Refactoring The interfaces of backtest and strategy module.


Version 0.4.4
--------------------
- Optimize cache generation performance
- Add report module
- Fix bug when using ``ServerDatasetCache`` offline.
- In the previous version of ``long_short_backtest``, there is a case of ``np.nan`` in long_short. The current version ``0.4.4`` has been fixed, so ``long_short_backtest`` will be different from the previous version.
- In the ``0.4.2`` version of ``risk_analysis`` function, ``N`` is ``250``, and ``N`` is ``252`` from ``0.4.3``, so ``0.4.2`` is ``0.002122`` smaller than the ``0.4.3`` the backtest result is slightly different between ``0.4.2`` and ``0.4.3``.
- refactor the argument of backtest function.
    - **NOTE**:
      - The default arguments of topk margin strategy is changed. Please pass the arguments explicitly if you want to get the same backtest result as previous version.
      - The TopkWeightStrategy is changed slightly. It will try to sell the stocks more than ``topk``.  (The backtest result of TopkAmountStrategy remains the same)
- The margin ratio mechanism is supported in the Topk Margin strategies.


Version 0.4.5
--------------------
- Add multi-kernel implementation for both client and server.
    - Support a new way to load data from client which skips dataset cache.
    - Change the default dataset method from single kernel implementation to multi kernel implementation.
- Accelerate the high frequency data reading by optimizing the relative modules.
- Support a new method to write config file by using dict.

Version 0.4.6
--------------------
- Some bugs are fixed
    - The default config in `Version 0.4.5` is not friendly to daily frequency data.
    - Backtest error in TopkWeightStrategy when `WithInteract=True`.


Version 0.5.0
--------------------
- First opensource version
    - Refine the docs, code
    - Add baselines
    - public data crawler
