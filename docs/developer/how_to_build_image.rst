.. _docker_image:

==================
Build Docker Image
==================

Dockerfile
==========

There is a **Dockerfile** file in the root directory of the project from which you can build the docker image. There are two build methods in Dockerfile to choose from.
When executing the build command, use the ``--build-arg`` parameter to control the image version. The ``--build-arg`` parameter defaults to ``yes``, which builds the ``stable`` version of the qlib image.

1.For the ``stable`` version, use ``pip install pyqlib`` to build the qlib image.

.. code-block:: bash

    docker build --build-arg IS_STABLE=yes -t <image name> -f ./Dockerfile .

.. code-block:: bash

    docker build -t <image name> -f ./Dockerfile .

2. For the ``nightly`` version, use current source code to build the qlib image.

.. code-block:: bash

    docker build --build-arg IS_STABLE=no -t <image name> -f ./Dockerfile .

Auto build of qlib images
=========================

1. There is a **build_docker_image.sh** file in the root directory of your project, which can be used to automatically build docker images and upload them to your docker hub repository(Optional, configuration required).

.. code-block:: bash

    sh build_docker_image.sh
    >>> Do you want to build the nightly version of the qlib image? (default is stable) (yes/no):
    >>> Is it uploaded to docker hub? (default is no) (yes/no):

2. If you want to upload the built image to your docker hub repository, you need to edit your **build_docker_image.sh** file first, fill in ``docker_user`` in the file, and then execute this file.

How to use qlib images
======================
1. Start a new Docker container

.. code-block:: bash

    docker run -it --name <container name> -v <Mounted local directory>:/app <image name>

2. At this point you are in the docker environment and can run the qlib scripts. An example:

.. code-block:: bash

    >>> python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
    >>> python qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

3. Exit the container

.. code-block:: bash

    >>> exit

4. Restart the container

.. code-block:: bash

    docker start -i -a <container name>

5. Stop the container

.. code-block:: bash

    docker stop -i -a <container name>

6. Delete the container

.. code-block:: bash

    docker rm <container name>

7. For more information on using docker see the `docker documentation <https://docs.docker.com/reference/cli/docker/>`_.
