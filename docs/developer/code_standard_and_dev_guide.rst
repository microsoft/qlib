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

   You can fix the bug by inputing the following code in the command line.

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


=================================
Development Guidance
=================================

As a developer, you often want make changes to `Qlib` and hope it would reflect directly in your environment without reinstalling it. You can install `Qlib` in editable mode with following command.
The `[dev]` option will help you to install some related packages when developing `Qlib` (e.g. pytest, sphinx)

.. code-block:: bash

    pip install -e .[dev]
