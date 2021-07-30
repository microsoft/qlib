.. _code_standard:

=================================
Code Standard
=================================

Docstring
=================================
Please use the Numpy Style.

Continuous Integration
=================================
Continuous Integration (CI) tools help you stick to the quality standards by running tests every time you push a new commit and reporting the results to a pull request. 

A common error is the mixed use of space and tab. You can fix the bug by inputing the following code in the command line.

.. code-block:: python

    pip install black
    python -m black . -l 120