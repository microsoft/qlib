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

A common error is the mixed use of space and tab. You can fix the bug by inputing the following code in the command line.

.. code-block:: python

    pip install black
    python -m black . -l 120
