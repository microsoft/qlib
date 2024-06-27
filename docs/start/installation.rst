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


Running ``build_docker_image.sh`` is the right choice when you want to create a docker image for ``Qlib``.
Of course, before running it, please open this file and make some necessary changes according to your docker hub account.
.. code-block:: bash

   #!/bin/bash

   # Build the Docker image
   sudo docker build -t qlib_image -f ./Dockerfile .

   # Log in to Docker Hub
   # If you are a new docker hub user, please verify your email address before proceeding with this step.
   sudo docker login

   # Tag the Docker image
   sudo docker tag qlib_image <Your docker hub username, not your email>/qlib_image:stable

   # Push the Docker image to Docker Hub
   sudo docker push <Your docker hub username, not your email>/qlib_image:stable
