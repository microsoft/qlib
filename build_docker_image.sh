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
