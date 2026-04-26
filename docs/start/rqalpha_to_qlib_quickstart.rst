RQAlpha To Qlib Quickstart
==========================

This guide shows the shortest practical path to reuse an existing RQAlpha bundle for Qlib-based A-share research.

It covers four stages:

1. Find the real RQAlpha bundle
2. Convert it into Qlib data
3. Train a simple LightGBM model
4. Try a minimal backtest


1. What We Verified
-------------------

In this workspace, the following path was confirmed to be a valid RQAlpha bundle:

.. code-block:: text

   C:\Users\Administrator\.rqalpha\bundle

The bundle contains:

- ``stocks.h5``
- ``indexes.h5``
- ``ex_cum_factor.h5``
- ``instruments.pk``
- ``trading_dates.npy``

The stock dataset contains daily A-share bars such as:

- ``datetime``
- ``open``
- ``close``
- ``high``
- ``low``
- ``volume``
- ``total_turnover``

Qlib cannot use this bundle format directly, so we added a bridge script that converts it into Qlib format.
The converter also exports Qlib's ``factor`` field from RQAlpha's ``ex_cum_factor.h5`` so daily backtests can use normal trade-unit rounding instead of falling back to adjusted-price mode.


2. New Helper Scripts
---------------------

This repo now includes:

- ``scripts/convert_rqalpha_bundle.py``
- ``examples/rqalpha_train_predict.py``
- ``examples/rqalpha_fast_train_predict.py``
- ``examples/rqalpha_lgbm_param_sweep.py``
- ``examples/rqalpha_minimal_backtest.py``

Their responsibilities are:

- ``convert_rqalpha_bundle.py``: convert RQAlpha bundle data into Qlib-ready CSV and/or Qlib binary data
- ``rqalpha_train_predict.py``: the simplest teaching version of ``init -> dataset -> train -> predict``
- ``rqalpha_fast_train_predict.py``: the faster full-market research version that avoids repeated data preparation
- ``rqalpha_lgbm_param_sweep.py``: compare several LightGBM presets after the basic training path works
- ``rqalpha_minimal_backtest.py``: run a minimal signal-driven backtest after training


3. Convert RQAlpha Bundle To Qlib Data
--------------------------------------

3.1 Export CSV only
~~~~~~~~~~~~~~~~~~~

Use this if you want to inspect the intermediate files:

.. code-block:: bash

   python scripts\convert_rqalpha_bundle.py export_csv ^
     --bundle_path C:\Users\Administrator\.rqalpha\bundle ^
     --output_dir D:\Stock\rqalpha_csv ^
     --instrument_type stocks

This creates one CSV per stock, with columns like:

- ``date``
- ``symbol``
- ``open``
- ``close``
- ``high``
- ``low``
- ``volume``
- ``vwap``
- ``factor``

Notes:

- The converter maps ``000001.XSHE`` to ``SZ000001``
- The converter maps ``600000.XSHG`` to ``SH600000``
- ``vwap`` is computed from ``total_turnover / volume`` when the source does not provide it directly
- Price fields are adjusted by ``factor`` and ``volume`` is adjusted inversely, matching Qlib's expected daily data convention

3.2 Export and dump to Qlib directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most useful command for normal usage:

.. code-block:: bash

   python scripts\convert_rqalpha_bundle.py export_and_dump_qlib ^
     --bundle_path C:\Users\Administrator\.rqalpha\bundle ^
     --csv_dir D:\Stock\rqalpha_csv_factor ^
     --qlib_dir D:\Stock\qlib_from_rqalpha_factor ^
     --instrument_type stocks

If you want a smaller trial run first:

.. code-block:: bash

   python scripts\convert_rqalpha_bundle.py export_and_dump_qlib ^
     --bundle_path C:\Users\Administrator\.rqalpha\bundle ^
     --csv_dir D:\Stock\rqalpha_csv_sample ^
     --qlib_dir D:\Stock\qlib_from_rqalpha_sample ^
     --instrument_type stocks ^
     --limit 100

After conversion, a valid Qlib directory should contain:

- ``calendars/``
- ``features/``
- ``instruments/``

For backtests, prefer a converted directory that contains ``factor.day.bin`` under each stock feature directory. Without ``factor``, Qlib will warn that trade unit ``100`` is not supported in adjusted-price mode.


4. Train And Predict
--------------------

Once the Qlib directory exists, you can validate the whole research chain without touching the backtest layer yet.

Recommended for full-market research:

.. code-block:: bash

   python examples\rqalpha_fast_train_predict.py ^
     --provider-uri D:\Stock\qlib_from_rqalpha_factor ^
     --train-start 2010-01-01 ^
     --train-end 2018-12-31 ^
     --valid-start 2019-01-01 ^
     --valid-end 2019-12-31 ^
     --test-start 2020-01-01 ^
     --test-end 2020-12-31 ^
     --device gpu

This script is better for repeated A-share research because it:

- avoids the extra ``prepare(train/valid/test)`` calls used only for printing shapes
- keeps the command surface small while still supporting ``--device cpu/gpu``
- prints stage timings so bottlenecks are easier to see

If you want the simplest teaching version instead, use:

.. code-block:: bash

   python examples\rqalpha_train_predict.py ^
     --provider-uri D:\Stock\qlib_from_rqalpha_factor ^
     --train-start 2010-01-01 ^
     --train-end 2018-12-31 ^
     --valid-start 2019-01-01 ^
     --valid-end 2019-12-31 ^
     --test-start 2020-01-01 ^
     --test-end 2020-12-31

What this script does:

1. ``qlib.init``
2. Build ``Alpha158``
3. Split train/valid/test periods
4. Train ``LGBModel``
5. Output prediction scores on the test segment

The faster script was verified in this workspace on both sample and full-market data.

The simpler teaching script was also verified earlier. A real run produced:

- ``train shape=(43737, 159)``
- ``valid shape=(4636, 159)``
- ``test shape=(4377, 159)``

And prediction output like:

.. code-block:: text

   datetime    instrument
   2020-01-02  SZ000001    -0.018893
               SZ000002     0.021801
               SZ000004     0.011163

That is the strongest confirmation that the converted RQAlpha data is already usable by Qlib for model research.

4.1 Compare LightGBM Parameter Presets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the fast training script works, use the parameter sweep script to compare a few reasonable LightGBM starting points:

.. code-block:: bash

   python examples\rqalpha_lgbm_param_sweep.py ^
     --provider-uri D:\Stock\qlib_from_rqalpha_factor ^
     --train-start 2010-01-01 ^
     --train-end 2018-12-31 ^
     --valid-start 2019-01-01 ^
     --valid-end 2019-12-31 ^
     --test-start 2020-01-01 ^
     --test-end 2020-12-31 ^
     --device gpu

This script prepares the Qlib dataset once, then trains multiple LightGBM configs on the same prepared arrays. It prints a CSV-style summary sorted by validation ``l2``.
For repeated research runs on the same date ranges, keep the default ``--dataset-cache simple`` so Qlib can reuse the prepared feature dataset locally.


5. Run A Minimal Backtest
-------------------------

After training and prediction work, try the minimal backtest script:

.. code-block:: bash

   python examples\rqalpha_minimal_backtest.py ^
     --provider-uri D:\Stock\qlib_from_rqalpha_factor ^
     --train-start 2010-01-01 ^
     --train-end 2018-12-31 ^
     --valid-start 2019-01-01 ^
     --valid-end 2019-12-31 ^
     --test-start 2020-01-01 ^
     --test-end 2020-12-31 ^
     --topk 5 ^
     --n-drop 1

This script does:

1. Train the model
2. Generate prediction scores
3. Feed them into ``TopkDropoutStrategy``
4. Run ``SimulatorExecutor``
5. Print backtest summary output

This is intended as the smallest end-to-end backtest example, not as a production strategy.

5.1 More Stable Research Backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tiny ``topk=5`` setup is useful for smoke tests, but it is too concentrated for judging A-share strategy quality.
Single-year parameter sweeps can also overfit. In this workspace, ``topk=30,n_drop=1`` was very strong in 2025 but failed in 2024.
For a more stable first research run, use ``topk=80,n_drop=10`` with a one-trading-day signal shift and a 252-day listing-history filter:

.. code-block:: bash

   python examples\rqalpha_lgbm_preset_backtest.py ^
     --provider-uri D:\Stock\qlib_from_rqalpha_factor ^
     --train-start 2010-01-01 ^
     --train-end 2018-12-31 ^
     --valid-start 2019-01-01 ^
     --valid-end 2019-12-31 ^
     --test-start 2020-01-01 ^
     --test-end 2020-12-31 ^
     --device gpu ^
     --presets baseline,wider ^
     --topk 80 ^
     --n-drop 10 ^
     --min-history-days 252 ^
     --signal-shift-days 1 ^
     --deal-price close

This command keeps the run fast enough for iteration while reducing the chance that a handful of extreme names dominate the result.
The ``--signal-shift-days 1`` option avoids using same-day close information to trade at the same close.
The research scripts now default to ``--device gpu`` and ``--dataset-cache simple``; on repeated runs, the cache can cut dataset initialization time dramatically.

5.2 GPU-Heavier Deep Learning Baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want a baseline that makes better use of a 16 GB GPU than LightGBM does, try the standalone MLP backtest script:

.. code-block:: bash

   python examples\rqalpha_dnn_backtest.py ^
     --provider-uri D:\Stock\qlib_from_rqalpha_factor ^
     --train-start 2022-01-01 ^
     --train-end 2022-12-31 ^
     --valid-start 2023-01-01 ^
     --valid-end 2023-06-30 ^
     --test-start 2023-07-01 ^
     --test-end 2023-09-30 ^
     --device gpu ^
     --layers 1024,512,256 ^
     --batch-size 8192 ^
     --topk 30 ^
     --n-drop 3

This script uses a PyTorch MLP instead of LightGBM, prints ``torch_cuda_max_memory_allocated_gb``, and reuses the same A-share signal filters and backtest settings as the LightGBM research workflow.


6. Recommended Learning Order
-----------------------------

If you are new to Qlib, use this order:

1. Run the converter on a small sample with ``--limit 100``
2. Run ``rqalpha_fast_train_predict.py``
3. Run ``rqalpha_lgbm_param_sweep.py`` to compare initial model settings
4. Confirm shapes and prediction output
5. Run ``rqalpha_minimal_backtest.py``
6. Only then move on to larger universes, longer periods, and custom strategies

This order matters because it isolates problems cleanly:

- Conversion issues stay in step 1
- Feature/model issues stay in step 2
- Strategy/backtest issues stay in step 4


7. Common Issues
----------------

Qlib cannot import local extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see errors related to:

- ``qlib.data._libs.rolling``
- ``qlib.data._libs.expanding``

build them in place:

.. code-block:: bash

   python setup.py build_ext --inplace

Missing optional packages during startup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depending on the exact path you run, Qlib may need packages such as:

- ``lightgbm``
- ``gym``
- ``cvxpy``
- ``mlflow``
- ``python-redis-lock``

The lightweight ``train_predict`` path is the best first validation because it proves the data and model path before you spend time on the heavier backtest stack.

Why the fast script is faster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original teaching script prepares ``train``, ``valid`` and ``test`` once for shape printing, and then the model prepares data again inside ``fit`` and ``predict``.

The faster script removes that duplicated preparation step, which was the most effective speedup in this workspace.

Why ``vwap`` was added
~~~~~~~~~~~~~~~~~~~~~~

``Alpha158`` uses price features including ``VWAP``, so the converter computes:

.. code-block:: text

   vwap = total_turnover / volume

If ``volume == 0``, it falls back to ``close``.


8. Summary
----------

The verified research path is:

.. code-block:: text

   RQAlpha bundle
   -> convert_rqalpha_bundle.py
   -> Qlib data directory
   -> rqalpha_fast_train_predict.py
   -> rqalpha_minimal_backtest.py

At this point, the important conclusion is already established:

``RQAlpha bundle data can be reused by Qlib for A-share model research after conversion.``
