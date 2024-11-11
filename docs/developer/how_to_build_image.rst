.. _docker_image:

=============
Docker Image
=============

Docstring
=========
Please use the `Numpydoc Style <https://stackoverflow.com/a/24385103>`_.

Continuous Integration
======================

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
   sudo docker tag qlib_image <Your docker hub username, not your email>/qlib_image:<version stable or nightly>

   # Push the Docker image to Docker Hub
   sudo docker push <Your docker hub username, not your email>/qlib_image:<version stable or nightly>


