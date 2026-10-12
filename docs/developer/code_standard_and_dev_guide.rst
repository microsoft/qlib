.. _code_standard:

=============
Code Standard
=============

Docstring
=========
Please use the `Numpydoc Style <https://stackoverflow.com/a/24385103>`_.

Continuous Integration
======================
Continuous Integration (CI) tools help you stick to the quality standards by running tests every time you push a new commit and reporting the results to a pull request.

When you submit a PR request, you can check whether your code passes the CI tests in the "check" section at the bottom of the web page.

1. Qlib will check the code format with black. The PR will raise error if your code does not align to the standard of Qlib(e.g. a common error is the mixed use of space and tab).

   You can fix the bug by inputting the following code in the command line.

.. code-block:: bash

    pip install black
    python -m black . -l 120


2. Qlib will check your code style pylint. The checking command is implemented in [github action workflow](https://github.com/microsoft/qlib/blob/0e8b94a552f1c457cfa6cd2c1bb3b87ebb3fb279/.github/workflows/test.yml#L66).
   Sometime pylint's restrictions are not that reasonable. You can ignore specific errors like this

.. code-block:: python

    return -ICLoss()(pred, target, index)  # pylint: disable=E1130


3. Qlib will check your code style flake8. The checking command is implemented in [github action workflow](https://github.com/microsoft/qlib/blob/0e8b94a552f1c457cfa6cd2c1bb3b87ebb3fb279/.github/workflows/test.yml#L73).

   You can fix the bug by inputing the following code in the command line.

.. code-block:: bash

    flake8 --ignore E501,F541,E402,F401,W503,E741,E266,E203,E302,E731,E262,F523,F821,F811,F841,E713,E265,W291,E712,E722,W293 qlib


4. Qlib has integrated pre-commit, which will make it easier for developers to format their code.

   Just run the following two commands, and the code will be automatically formatted using black and flake8 when the git commit command is executed.

.. code-block:: bash

    pip install -e .[dev]
    pre-commit install


Engineering Elevation Gates
===========================

In addition to the legacy black/pylint/flake8/mypy checks, new hardened modules
are gated by the additive ``engineering-elevation`` workflow
(``.github/workflows/engineering-elevation.yml``, Python 3.9–3.11):

- ``ruff check`` plus ``ruff format --check`` over the hardened modules;
- ``mypy --strict`` scoped to ``qlib/backtest/validation.py``,
  ``qlib/backtest/config_schema.py``, ``qlib/backtest/executor_logging.py``,
  ``qlib/backtest/data_provider.py``, ``qlib/log_structured.py``, ``qlib/data/cache_guards.py``,
  ``qlib/contrib/benchmark/base.py``, ``qlib/contrib/rl/config_schema.py``,
  ``qlib/utils/credentials.py``, ``qlib/contrib/data/processor_tracing.py``, and
  ``qlib/workflow/task/batch_limits.py``;
- filesystem-free tests in ``tests/backtest/test_backtest_core.py``,
  ``tests/backtest/test_elevation_phase2.py``,
  ``tests/backtest/test_elevation_phase3.py``,
  ``tests/backtest/test_elevation_phase4.py``, and
  ``tests/backtest/test_exchange_with_provider.py`` with a ``pytest-cov`` 80% gate;
- schema validation of ``examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml``
  and ``examples/rl_order_execution/exp_configs/backtest_twap.yml``.

Packaging standards (Ruff/Black/MyPy/Coverage sections) live in
``pyproject.toml``; local parity with the container is via ``.env.example``
(copy to ``.env``, never commit secrets) and the multi-stage ``Dockerfile``.
Architecture rationale is recorded in ``docs/adr/0001-backtest-boundaries.md``.


=================================
Development Guidance
=================================

As a developer, you often want make changes to `Qlib` and hope it would reflect directly in your environment without reinstalling it. You can install `Qlib` in editable mode with following command.
The `[dev]` option will help you to install some related packages when developing `Qlib` (e.g. pytest, sphinx)

.. code-block:: bash

    pip install -e ".[dev]"