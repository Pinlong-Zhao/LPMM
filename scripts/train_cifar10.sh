#!/bin/bash

# Training script for CIFAR-10

# Set default values
IPC=${1:-10}
GPU=${2:-0}

echo "Training LPMM on CIFAR-10 with IPC=$IPC on GPU $GPU"

python main.py \
    --config config/cifar10.yaml \
    --ipc $IPC \
    --gpu $GPU \
    --mode train

echo "Training completed!"
