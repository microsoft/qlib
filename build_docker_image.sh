#!/bin/bash

# Build the Docker image
sudo docker build -t qlib_image -f ./Dockerfile .

# Log in to Docker Hub
sudo docker login

# Tag the Docker image
sudo docker tag qlib_image linlanglv/qlib_image:stable

# Push the Docker image to Docker Hub
sudo docker push linlanglv/qlib_image:stable
