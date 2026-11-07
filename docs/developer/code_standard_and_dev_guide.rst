.. _code_standard:

=============
Code Standard
=============

Docstring
=========
Please use the `Numpydoc Style <https://stackoverflow.com/a/24385103>`_.

Continuous Integration
======================
Qlib runs formatting and Python-style checks in the
`source test workflow <https://github.com/microsoft/qlib/blob/main/.github/workflows/test_qlib_from_source.yml>`_.
The workflow uses the lint targets in the root ``Makefile``. Install their
dependencies before running the same checks locally:

.. code-block:: bash

    python -m pip install -e ".[lint]"

1. Run ``make black`` to check formatting. To format a changed file, run
   ``black <file> -l 120`` and repeat ``make black``.

2. Run ``make pylint`` to check Python code style. When a specific rule needs
   an exception, document it at the relevant line, for example:

   .. code-block:: python

      return -ICLoss()(pred, target, index)  # pylint: disable=E1130

3. Run ``make flake8`` to check flake8 rules. The Makefile target uses the
   same options as CI.

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

    pip install -e ".[dev]"