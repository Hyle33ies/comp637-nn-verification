#!/usr/bin/env bash

# Create result and log directories
mkdir -p ./results/cifar_atas_resnet18
mkdir -p ./log

# --- Model & Training Configuration --- 
# Basic settings
DATASET="cifar10"                  # Dataset name (cifar10 or cifar100)
ARCH="ResNet18"                    # Model architecture
EPOCHS=50                          # Total training epochs
BATCH_SIZE=64                     # Training batch size (Increased to standard 128)
TEST_BATCH_SIZE=128                # Evaluation batch size
LR=0.1                             # Initial learning rate (Adjusted for BS=128)
WEIGHT_DECAY=5e-4                  # Weight decay for regularization
MOMENTUM=0.9                       # SGD momentum

# Learning rate schedule
# Note: Learning rate will be reduced by 10x at these epochs
DECAY_STEPS="15 25"                # Adjusted decay points here

# Adversarial training parameters
EPSILON=8                          # Perturbation size (8/255)
WARMUP_EPOCHS=4                   # Natural training epochs before adversarial training

# ATAS specific parameters
C=0.01                             # Hard fraction for adaptive step size
MAX_STEP_SIZE=12                   # Maximum perturbation step size
MIN_STEP_SIZE=4                    # Minimum perturbation step size
EPOCHS_RESET=4                     # Reset perturbations

# Output directories & resources
MODEL_DIR="./results/cifar_atas_resnet18"
mkdir -p $MODEL_DIR
LOG_DIR="./log"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/cifar_atas_resnet18.log"
NUM_WORKERS=4

# Display configuration
echo "==== CIFAR Adversarial Training ====="
echo "Dataset: $DATASET"
echo "Model: $ARCH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Epsilon: $EPSILON/255"
echo "Results will be saved to: $MODEL_DIR"
echo "Log file: $LOG_FILE"
echo "======================================"

# Run ATAS adversarial training
python3 -u ATAS.py \
  --dataset $DATASET \
  --arch $ARCH \
  --batch-size $BATCH_SIZE \
  --test-batch-size $TEST_BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --weight-decay $WEIGHT_DECAY \
  --momentum $MOMENTUM \
  --decay-steps $DECAY_STEPS \
  --epsilon $EPSILON \
  --max-step-size $MAX_STEP_SIZE \
  --min-step-size $MIN_STEP_SIZE \
  --c $C \
  --model-dir $MODEL_DIR \
  --num-workers $NUM_WORKERS \
  --warmup-epochs $WARMUP_EPOCHS \
  --epochs-reset $EPOCHS_RESET \
  | tee $LOG_FILE

echo "ATAS training completed! Results saved to $MODEL_DIR"

# Final attack evaluation
echo "Running final attack evaluation..."
ATTACK_LOG="${LOG_DIR}/attack_${DATASET}_${ARCH}_eps${EPSILON}.log"
python3 -u attack.py \
  --dataset $DATASET \
  --model-dir $MODEL_DIR \
  --arch $ARCH \
  --epsilon $EPSILON > $ATTACK_LOG 2>&1

echo "Attack evaluation completed! Results saved to $ATTACK_LOG"

# Copy the model to the verification location
# VERIFY_MODEL_DIR="../alpha-beta-CROWN/complete_verifier/models/cifar10_resnet"
# echo "Copying model to verification directory..."
# mkdir -p $VERIFY_MODEL_DIR
# cp $MODEL_DIR/best.pth $VERIFY_MODEL_DIR/
# echo "Model copied to $VERIFY_MODEL_DIR/best.pth"
