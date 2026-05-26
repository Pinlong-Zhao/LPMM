#!/bin/bash

# Evaluation script

CONFIG=${1:-config/cifar10.yaml}
CHECKPOINT=${2:-./checkpoints/best_model.pth}
GPU=${3:-0}

echo "Evaluating model from $CHECKPOINT"

python main.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --gpu $GPU \
    --mode eval

echo "Evaluation completed!"
