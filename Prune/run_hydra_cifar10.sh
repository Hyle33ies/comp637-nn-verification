#!/bin/bash

# Script to run HYDRA for CIFAR-10 with WRN-28-4 model and 90% sparsity
# Based on the original paper: "HYDRA: Pruning Adversarially Robust Neural Networks"

# Create directories
mkdir -p trained_models

# Set parameters
ARCH="wrn_28_4"
EXP_NAME="cifar10-$ARCH-adv-k_0.1"
TRAINER="adv"      # Use adversarial training (TRADES)
VAL_METHOD="adv"   # Evaluate with adversarial examples
GPU_ID="0"         # GPU to use
K=0.1              # Keep 10% of weights (90% sparsity)
# The following are the default values for the hyperparameters as in configs/configs.yml
PRETRAIN_EPOCHS=100 # Epochs for pretraining phase
PRUNE_EPOCHS=20    # Epochs for pruning phase
FINETUNE_EPOCHS=100 # Epochs for finetuning phase

# Create log directory
LOG_DIR="./trained_models/${EXP_NAME}/logs"
mkdir -p ${LOG_DIR}

echo "Starting HYDRA pipeline for CIFAR-10 with $ARCH (90% pruning)"
echo "=================================================="

# Step 1: Pre-training
echo "Step 1/3: Pre-training with adversarial training"
python train.py \
    --exp-name $EXP_NAME \
    --arch $ARCH \
    --exp-mode pretrain \
    --configs configs/configs.yml \
    --trainer $TRAINER \
    --val_method $VAL_METHOD \
    --gpu $GPU_ID \
    --k 1.0 \
    --epochs $PRETRAIN_EPOCHS \
    --save-dense \
    | tee ${LOG_DIR}/pretrain.log

# Step 2: Pruning
echo "Step 2/3: Pruning with adversarial training"
python train.py \
    --exp-name $EXP_NAME \
    --arch $ARCH \
    --exp-mode prune \
    --configs configs/configs.yml \
    --trainer $TRAINER \
    --val_method $VAL_METHOD \
    --gpu $GPU_ID \
    --k $K \
    --save-dense \
    --scaled-score-init \
    --source-net ./trained_models/$EXP_NAME/pretrain/latest_exp/checkpoint/checkpoint.pth.tar \
    --epochs $PRUNE_EPOCHS \
    | tee ${LOG_DIR}/prune.log

# Step 3: Fine-tuning
echo "Step 3/3: Fine-tuning of pruned network"
python train.py \
    --exp-name $EXP_NAME \
    --arch $ARCH \
    --exp-mode finetune \
    --configs configs/configs.yml \
    --trainer $TRAINER \
    --val_method $VAL_METHOD \
    --gpu $GPU_ID \
    --k $K \
    --save-dense \
    --source-net ./trained_models/$EXP_NAME/prune/latest_exp/checkpoint/checkpoint.pth.tar \
    --lr 0.01 \
    --epochs $FINETUNE_EPOCHS \
    | tee ${LOG_DIR}/finetune.log

echo "HYDRA training complete!"
echo "Results saved in: ./trained_models/$EXP_NAME"
echo "Logs saved in: ${LOG_DIR}" 
