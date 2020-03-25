#!/bin/bash


HERE=$(pwd) # Absolute path of current directory
user=`whoami`
uid=`id -u`
gid=`id -g`


# get the latest nightly build
docker pull tensorflow/tensorflow:nightly-gpu-py3

# get the latest official release
#docker pull tensorflow/tensorflow:latest-gpu-py3


# current directory will be '/workspace'
docker run --gpus all --privileged=true -it --rm \
           -u $(id -u):$(id -g) \
           -e USER=$user -e UID=$uid -e GID=$gid \
           -w /workspace \
           -v $HERE:/workspace \
           --network=host \
           tensorflow/tensorflow:nightly-gpu-py3 \
           bash
