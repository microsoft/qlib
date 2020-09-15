QLib
==========

QLib is a Quantitative-research Library, which can provide research data with highly consistency, reusability and extensibility.

.. note:: Anaconda python is strongly recommended for this library. See https://www.anaconda.com/download/.


Install
----------

Install as root:

.. code-block:: bash

   $ python setup.py install

   
Install as single user (if you have no root permission):

.. code-block:: bash

   $ python setup.py install --user


To verify your installation, open your python shell:

.. code-block:: python

   >>> import qlib
   >>> qlib.__version__
   '0.2.2'

You can also run ``tests/test_data_sim.py`` to verify your installation.
