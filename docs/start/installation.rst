.. _installation:

============
Installation
============

.. currentmodule:: qlib


``Qlib`` Installation
=====================
.. note::

   `Qlib` supports both `Windows` and `Linux`. It's recommended to use `Qlib` in `Linux`. ``Qlib`` supports Python3, which is up to Python3.8.

Users can easily install ``Qlib`` by pip according to the following command:

.. code-block:: bash

   pip install pyqlib


Also, Users can install ``Qlib`` by the source code according to the following steps:

- Enter the root directory of ``Qlib``, in which the file ``setup.py`` exists.
- Then, please execute the following command to install the environment dependencies and install ``Qlib``:

   .. code-block:: bash

      $ pip install numpy
      $ pip install --upgrade cython
      $ git clone https://github.com/microsoft/qlib.git && cd qlib
      $ python setup.py install

.. note::
   It's recommended to use anaconda/miniconda to setup the environment. ``Qlib`` needs lightgbm and pytorch packages, use pip to install them.



Use the following code to make sure the installation successful:

.. code-block:: python

   >>> import qlib
   >>> qlib.__version__
   <LATEST VERSION>
