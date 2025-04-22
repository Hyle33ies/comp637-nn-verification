#!/bin/bash

# Script to run HYDRA pruning (stage 2) on GTSRB using a pretrained model

# --- Configuration ---
PRETRAINED_MODEL="./pretrained/best.pt" # Path to the adversarially trained model
CONFIG_FILE="configs/gtsrb_configs.yml" # Path to the GTSRB pruning config
ARCH="wrn_28_4_prune"
PRUNE_RATIO=0.05 # Target prune ratio (k = 1 - sparsity, 0.05 means 95% sparse)
EXP_NAME="gtsrb_hydra_prune_wrn28_4_95" 
EPOCHS=20 
GPU_ID="0"  
#---------------------

# --- Safety Checks ---
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Error: Pretrained model not found at $PRETRAINED_MODEL" 
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# --- Create Result Directory (using exp_name from config) ---
# Ensure the base result directory exists (adjust if different in config)
BASE_RESULT_DIR="./trained_models"
mkdir -p "${BASE_RESULT_DIR}/${EXP_NAME}"

# --- Run Pruning ---
echo "Starting HYDRA pruning for GTSRB..."
python train.py \
    --configs ${CONFIG_FILE} \
    --exp-mode prune \
    --arch ${ARCH} \
    --k ${PRUNE_RATIO} \
    --scaled-score-init \
    --source-net ${PRETRAINED_MODEL} \
    --save-dense \
    --epochs ${EPOCHS} \
    --gpu ${GPU_ID} \
    | tee "${BASE_RESULT_DIR}/${EXP_NAME}/pruning_log.txt"

echo "HYDRA pruning completed! Results in ${BASE_RESULT_DIR}/${EXP_NAME}" 
