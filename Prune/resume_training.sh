#!/bin/bash

# Script to resume HYDRA training from a checkpoint
# Usage: ./resume_training.sh [phase] [epoch]
# Example: ./resume_training.sh pretrain 50

# Validate arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: ./resume_training.sh [phase] [epoch]"
    echo "  phase: pretrain, prune, or finetune"
    echo "  epoch: (optional) epoch number to resume from (default: latest)"
    exit 1
fi

# Set parameters
PHASE=$1
ARCH="wrn_28_4"
EXP_NAME="cifar10-$ARCH-adv-k_0.1"
TRAINER="adv"
VAL_METHOD="adv"
GPU_ID="0"
K=0.1  # 90% sparsity

# Create log directory if it doesn't exist
LOG_DIR="./trained_models/${EXP_NAME}/logs"
mkdir -p ${LOG_DIR}

# Determine the checkpoint to resume from
CHECKPOINT_DIR="./trained_models/${EXP_NAME}/${PHASE}/latest_exp/checkpoint"

if [ "$#" -eq 2 ]; then
    # If epoch is specified, construct path to that specific checkpoint
    EPOCH=$2
    CHECKPOINT="${CHECKPOINT_DIR}/epoch_${EPOCH}.pth.tar"
    
    # Check if the specific epoch checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint for epoch ${EPOCH} not found at ${CHECKPOINT}"
        echo "Using the latest checkpoint instead."
        CHECKPOINT="${CHECKPOINT_DIR}/checkpoint.pth.tar"
    fi
else
    # Use the latest checkpoint
    CHECKPOINT="${CHECKPOINT_DIR}/checkpoint.pth.tar"
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

echo "Resuming ${PHASE} from checkpoint: ${CHECKPOINT}"

# Set phase-specific parameters
case $PHASE in
    pretrain)
        EPOCHS=100
        LR=0.1
        ;;
    prune)
        EPOCHS=20
        LR=0.01
        SOURCE_NET="--source-net ./trained_models/${EXP_NAME}/pretrain/latest_exp/checkpoint/checkpoint.pth.tar"
        EXTRA_ARGS="--scaled-score-init"
        ;;
    finetune)
        EPOCHS=100
        LR=0.01
        SOURCE_NET="--source-net ./trained_models/${EXP_NAME}/prune/latest_exp/checkpoint/checkpoint.pth.tar"
        ;;
    *)
        echo "Error: Invalid phase. Use 'pretrain', 'prune', or 'finetune'."
        exit 1
        ;;
esac

# Resume training
echo "Resuming ${PHASE} phase (HYDRA training for CIFAR-10 with ${ARCH})"
echo "=================================================="

python train.py \
    --exp-name $EXP_NAME \
    --arch $ARCH \
    --exp-mode $PHASE \
    --configs configs/configs.yml \
    --trainer $TRAINER \
    --val_method $VAL_METHOD \
    --gpu $GPU_ID \
    --k ${K} \
    --save-dense \
    --resume ${CHECKPOINT} \
    --epochs $EPOCHS \
    --lr $LR \
    ${SOURCE_NET} \
    ${EXTRA_ARGS} \
    | tee -a ${LOG_DIR}/${PHASE}_resumed.log

echo "Resume complete!"
echo "Results saved in: ./trained_models/$EXP_NAME"
echo "Logs saved in: ${LOG_DIR}" 
