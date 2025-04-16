#!/bin/bash

# Create results directory
mkdir -p ./results/gtsrb_natural

# Run fine-tuning
python finetune_gtsrb.py \
    --batch-size 64 \
    --test-batch-size 128 \
    --epochs 10 \
    --lr 0.001 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --num-workers 12 \
    --seed 1 \
    --model-dir ./results/gtsrb 
