#!/bin/bash


# train, evaluate and save keras model
train() {
  python train.py \
    --opt          rms \
    --epochs       140 \
    --learnrate    0.001 \
    --batchsize    125 \
    --tboard       ./tb_log \
    --keras_hdf5   ./densenet.h5
}

train 2>&1 | tee ./train_log.txt



