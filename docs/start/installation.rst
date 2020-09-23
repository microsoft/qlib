.. _installation:
====================
Installation
====================

.. currentmodule:: qlib


How to Install ``Qlib``
====================

``Qlib`` only supports Python3.
.. and supports up to Python3.8.

Please follow the steps blow to install ``Qlib``:

- Change the directory to ``Qlib``, in which the file ``setup.py`` exists.
- Then, please execute the following command:
   
   .. code-block:: bash

      $ pip install numpy
      $ pip install --upgrade cython
      $ git clone https://github.com/microsoft/qlib.git && cd qlib
      $ python setup.py install


.. note::
   It's recommended to use anaconda/miniconda to setup environment.
   ``Qlib`` needs lightgbm and pytorch packages, use pip to install them.

.. note::
   Do not import qlib in the repository folder which contains ``qlib``, otherwise errors may occur.
   


Use the following code to make sure the installation is successful:

.. code-block:: python

   >>> import qlib
   >>> qlib.__version__
   <LATEST VERSION>


