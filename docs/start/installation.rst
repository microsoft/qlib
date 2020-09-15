.. _installation:
====================
Installation
====================

.. currentmodule:: qlib


How to Install Qlib
====================

``Qlib`` only supports Python3, and supports up to Python3.8.

Please execute the following process to install ``Qlib``:

- Change the directory to Qlib, and the file'setup.py' exists in the directory
- Then, execute the following command:
   
   .. code-block:: bash

      $ pip install numpy
      $ pip install --upgrade cython
      $ python setup.py install


.. note::
   It's recommended to use anaconda/miniconda to setup environment.
   ``Qlib`` needs lightgbm and tensorflow packages, use pip to install them.

.. note::
   Do not import qlib in the ``Qlib`` folder, otherwise errors may occur.
   


Use the following code to confirm installation successful:

.. code-block:: python

   >>> import qlib
   >>> qlib.__version__
   <LATEST VERSION>

..
   .. note:: Please read this documentation carefully since there are lots of changes in qlib.

..
   .. note:: On client side, there are some configs you need to notice like the providers, flask_server, flask_port and mount_path. The default is built for 10.150.144.153 since the server data path is pre-mounted to the mount_path. Don't change these configs unless you have some special test purposes.


..
   .. note:: You can always refer to the server docs on http://10.150.144.154:10002




