#!/bin/bash

# Script to run HYDRA fine-tuning (stage 3) on GTSRB using a pruned model

# --- Configuration ---
PRUNED_MODEL="./trained_models/gtsrb_hydra_prune_wrn28_4_80/prune/latest_exp/checkpoint/model_best.pth.tar"
CONFIG_FILE="configs/gtsrb_configs.yml"        # Base config file (lr/epochs will be overridden)
ARCH="wrn_28_4_prune"                      # MUST match the pruned model architecture
PRUNE_RATIO=0.2                             # MUST match the pruning sparsity level (k)
EXP_NAME="gtsrb_hydra_finetune_wrn28_4_80"           # Experiment name for fine-tuning logs/results
EPOCHS=30                                    # Fine-tuning epochs (adjust as needed)
LR=0.001                                     # Fine-tuning learning rate (adjusted lower)
GPU_ID="0"                                   # GPU to use
#---------------------

# --- Safety Checks ---
if [ ! -f "$PRUNED_MODEL" ]; then
    echo "Error: Pruned model checkpoint not found at $PRUNED_MODEL" 
    echo "Please verify the path and ensure the pruning stage completed successfully."
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# --- Create Result Directory ---
# Construct the results directory path based on the experiment name
FINETUNE_RESULT_DIR="./trained_models/${EXP_NAME}"
mkdir -p "${FINETUNE_RESULT_DIR}"

# --- Run Fine-tuning ---
echo "Starting HYDRA fine-tuning for GTSRB..."
python train.py \
    --arch ${ARCH} \
    --configs ${CONFIG_FILE} \
    --exp-mode finetune \
    --trainer adv \
    --val_method adv \
    --source-net ${PRUNED_MODEL} \
    --save-dense \
    --lr ${LR} \
    --k ${PRUNE_RATIO} \
    --epochs ${EPOCHS} \
    --gpu ${GPU_ID} \
    --exp-name ${EXP_NAME} \
    | tee "${FINETUNE_RESULT_DIR}/finetuning_log.txt"

# Note: --scaled-score-init is NOT used here, as scores are loaded from the pruned checkpoint.

echo "HYDRA fine-tuning completed! Results in ${FINETUNE_RESULT_DIR}"
