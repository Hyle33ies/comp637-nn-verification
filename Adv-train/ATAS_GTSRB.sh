#!/bin/bash

# Create result and log directories
mkdir -p ./results/gtsrb_atas_wrn28_4
mkdir -p ./log

# --- Model & Training Configuration --- 
# Basic settings
MODEL_NAME="WideResNet-16-2"      # Model name for logging
BATCH_SIZE=64                      # Training batch size
TEST_BATCH_SIZE=128                # Evaluation batch size
EPOCHS=30                          # Total training epochs
LR=0.01                            # Initial learning rate
WEIGHT_DECAY=5e-4                  # Weight decay for regularization
MOMENTUM=0.9                       # SGD momentum

# Learning rate schedule
# Note: Learning rate will be reduced by 10x at these epochs
DECAY_STEPS="24 28"

# Adversarial training parameters
EPSILON=8                          # Perturbation size (8/255)
WARMUP_EPOCHS=3                    # Natural training epochs before adversarial training
DROPOUT=0.2

# ATAS specific parameters
C=0.01                             # Hard fraction for adaptive step size
MAX_STEP_SIZE=14                   # Maximum perturbation step size
MIN_STEP_SIZE=4                    # Minimum perturbation step size
EPOCHS_RESET=4                    # Reset perturbations every N epochs

# Output directories & resources
MODEL_DIR=./results/gtsrb_atas_wrn16_2
LOG_FILE=log/atas_gtsrb_wrn16_2.log
NUM_WORKERS=6                      # Number of data loader workers, set to CPU cores

# Display configuration
echo "==== GTSRB Adversarial Training ====="
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Epsilon: $EPSILON/255"
echo "Results will be saved to: $MODEL_DIR"
echo "Log file: $LOG_FILE"
echo "======================================"

# Run ATAS adversarial training on GTSRB
python ATAS_GTSRB.py \
  --batch-size ${BATCH_SIZE} \
  --test-batch-size ${TEST_BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --momentum ${MOMENTUM} \
  --decay-steps ${DECAY_STEPS} \
  --epsilon ${EPSILON} \
  --dropout-rate ${DROPOUT} \
  --max-step-size ${MAX_STEP_SIZE} \
  --min-step-size ${MIN_STEP_SIZE} \
  --c ${C} \
  --model-dir ${MODEL_DIR} \
  --num-workers ${NUM_WORKERS} \
  --warmup-epochs ${WARMUP_EPOCHS} \
  --epochs-reset ${EPOCHS_RESET} \
  --arch WideResNet \
  --depth 16 \
  --widen-factor 2 | tee ${LOG_FILE}

echo "ATAS training completed! Results saved to ${MODEL_DIR}" 
