.. _code_standard:

=================================
Code Standard
=================================

Docstring
=================================
Please use the `Numpydoc Style <https://stackoverflow.com/a/24385103>`_.

Continuous Integration
=================================
Continuous Integration (CI) tools help you stick to the quality standards by running tests every time you push a new commit and reporting the results to a pull request. 

When you submit a PR request, you can check whether your code passes the CI tests in the "check" section at the bottom of the web page.

1. Qlib will check the code format with black. The PR will raise error if your code does not align to the standard of Qlib(e.g. a common error is the mixed use of space and tab).
 You can fix the bug by inputing the following code in the command line.

.. code-block:: python

    pip install black
    python -m black . -l 120


2. Qlib will check your code style pylint. The checking command is implemented in [github action workflow](https://github.com/microsoft/qlib/blob/0e8b94a552f1c457cfa6cd2c1bb3b87ebb3fb279/.github/workflows/test.yml#L66). 
   Sometime pylint's restrictions are not that reasonable. You can ignore specific errors like this

.. code-block:: python

    return -ICLoss()(pred, target, index)  # pylint: disable=E1130

