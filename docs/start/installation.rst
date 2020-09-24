.. _installation:
====================
Installation
====================

.. currentmodule:: qlib


``Qlib`` Installation
=====================
.. note::

   `Qlib` supports both `Windows` and `Linux`. It's recommended to use `Qlib` in `Linux`. ``Qlib`` supports Python3, which is up to Python3.8.

Please follow the steps below to install ``Qlib``:

- Enter the root directory of ``Qlib``, in which the file ``setup.py`` exists.
- Then, please execute the following command to install the environment dependencies and install ``Qlib``:
   
   .. code-block:: bash

      $ pip install numpy
      $ pip install --upgrade cython
      $ git clone https://github.com/microsoft/qlib.git && cd qlib
      $ python setup.py install


.. note::
   It's recommended to use anaconda/miniconda to setup the environment. ``Qlib`` needs lightgbm and pytorch packages, use pip to install them.

.. note::
   Do not import qlib in the root directory of ``Qlib``, otherwise, errors may occur.
   


Use the following code to make sure the installation successful:

.. code-block:: python

   >>> import qlib
   >>> qlib.__version__
   <LATEST VERSION>


=====================
