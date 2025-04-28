#!/usr/bin/env bash

# Create result and log directories
mkdir -p ./results/cifar_atas_resnet4b
mkdir -p ./results/cifar_atas_resnet4b_diffusion
mkdir -p ./log

# --- Model & Training Configuration --- 
# Basic settings
DATASET="cifar10"                  # Dataset name (cifar10 only)
ARCH="resnet4b"                    # Model architecture (smaller ResNet variant)
EPOCHS=200                          # Total training epochs (reduced for smaller model)
BATCH_SIZE=64                     # Training batch size
TEST_BATCH_SIZE=128                # Evaluation batch size
LR=0.05                            # Initial learning rate (reduced for smaller model)
WEIGHT_DECAY=5e-4                  # Weight decay for regularization
MOMENTUM=0.9                       # SGD momentum

# Learning rate schedule
# Note: Learning rate will be reduced by 10x at these epochs
DECAY_STEPS="40 100"                # Adjusted decay points for the smaller model

# Adversarial training parameters
EPSILON=4                          # Perturbation size (8/255)
WARMUP_EPOCHS=30                    # Natural training epochs before adversarial training

# ATAS specific parameters
C=0.01                             # Hard fraction for adaptive step size
MAX_STEP_SIZE=12                   # Maximum perturbation step size
MIN_STEP_SIZE=4                    # Minimum perturbation step size
EPOCHS_RESET=5                     # Reset perturbations

# Synthetic data parameters
SYNTHETIC_DATA_PATH="../1m.npz"     # Path to synthetic data file
REAL_SAMPLES=45000                 # Number of real samples to use per epoch (all of CIFAR-10)
SYNTHETIC_SAMPLES=45000            # Number of synthetic samples to use per epoch (1:1 ratio)

# Output directories & resources
MODEL_DIR="./results/cifar_atas_resnet4b"
DIFFUSION_MODEL_DIR="./results/cifar_atas_resnet4b_diffusion"
mkdir -p $MODEL_DIR
mkdir -p $DIFFUSION_MODEL_DIR
LOG_DIR="./log"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/cifar_atas_resnet4b.log"
DIFFUSION_LOG_FILE="${LOG_DIR}/cifar_atas_resnet4b_diffusion.log"
NUM_WORKERS=4                      # Reduced for smaller model

# Display standard training configuration
# echo "==== CIFAR Adversarial Training (Small Model) ====="
# echo "Dataset: $DATASET"
# echo "Model: $ARCH"
# echo "Batch size: $BATCH_SIZE"
# echo "Epochs: $EPOCHS"
# echo "Learning rate: $LR"
# echo "Epsilon: $EPSILON/255"
# echo "Results will be saved to: $MODEL_DIR"
# echo "Log file: $LOG_FILE"
# echo "=================================================="

# Run ATAS adversarial training with standard CIFAR-10
# python3 -u ATAS.py \
#   --dataset $DATASET \
#   --arch $ARCH \
#   --batch-size $BATCH_SIZE \
#   --test-batch-size $TEST_BATCH_SIZE \
#   --epochs $EPOCHS \
#   --lr $LR \
#   --weight-decay $WEIGHT_DECAY \
#   --momentum $MOMENTUM \
#   --decay-steps $DECAY_STEPS \
#   --epsilon $EPSILON \
#   --max-step-size $MAX_STEP_SIZE \
#   --min-step-size $MIN_STEP_SIZE \
#   --c $C \
#   --model-dir $MODEL_DIR \
#   --num-workers $NUM_WORKERS \
#   --warmup-epochs $WARMUP_EPOCHS \
#   --epochs-reset $EPOCHS_RESET \
#   | tee $LOG_FILE

# echo "ATAS training completed! Results saved to $MODEL_DIR"

# Display diffusion-augmented training configuration
echo ""
echo "==== CIFAR Adversarial Training with Diffusion Data ====="
echo "Dataset: $DATASET"
echo "Model: $ARCH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Epsilon: $EPSILON/255"
echo "Warmup epochs: $WARMUP_EPOCHS"
echo "Real samples per epoch: $REAL_SAMPLES"
echo "Synthetic samples per epoch: $SYNTHETIC_SAMPLES"
echo "Resampling synchronized with perturbation reset every $EPOCHS_RESET epochs"
echo "Results will be saved to: $DIFFUSION_MODEL_DIR"
echo "Log file: $DIFFUSION_LOG_FILE"
echo "=================================================="

# Run ATAS adversarial training with diffusion-augmented CIFAR-10
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
  --model-dir $DIFFUSION_MODEL_DIR \
  --num-workers $NUM_WORKERS \
  --warmup-epochs $WARMUP_EPOCHS \
  --epochs-reset $EPOCHS_RESET \
  --use-synthetic-data \
  --synthetic-data-path $SYNTHETIC_DATA_PATH \
  --real-samples $REAL_SAMPLES \
  --synthetic-samples $SYNTHETIC_SAMPLES \
  --progressive-mixing \
  --consistent-sampling \
  --filter-synthetic \
  --sync-resample \
  | tee $DIFFUSION_LOG_FILE

echo "ATAS training with diffusion data completed! Results saved to $DIFFUSION_MODEL_DIR"

# Final attack evaluation for diffusion-augmented model
# echo "Running final attack evaluation for diffusion-augmented model..."
# DIFFUSION_ATTACK_LOG="${LOG_DIR}/attack_diffusion_${DATASET}_${ARCH}_eps${EPSILON}.log"
# python3 -u attack.py \
#   --dataset $DATASET \
#   --model-dir $DIFFUSION_MODEL_DIR \
#   --arch $ARCH \
#   --epsilon $EPSILON > $DIFFUSION_ATTACK_LOG 2>&1

# echo "Attack evaluation for diffusion-augmented model completed! Results saved to $DIFFUSION_ATTACK_LOG"
