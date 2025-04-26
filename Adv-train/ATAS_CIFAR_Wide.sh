#!/usr/bin/env bash

# Create result and log directories
mkdir -p ./results/cifar_atas_resnet4b_ultrawide
mkdir -p ./log

# --- Model & Training Configuration --- 
# Basic settings
DATASET="cifar10"                  # Dataset name (cifar10 only)
ARCH="resnet4b_ultrawide"          # Model architecture (ultra-wide ResNet variant with in_planes=64)
EPOCHS=200                         # Total training epochs (increased for better convergence)
BATCH_SIZE=64                      # Training batch size
TEST_BATCH_SIZE=128                # Evaluation batch size
LR=0.1                             # Initial learning rate
WEIGHT_DECAY=7e-4                  # Weight decay for regularization
MOMENTUM=0.9                       # SGD momentum

# Learning rate schedule
# Note: Learning rate will be reduced by 10x at these epochs
DECAY_STEPS="40 80"             # Adjusted decay points for the wider model

# Adversarial training parameters
EPSILON=8                          # Perturbation size (8/255)
WARMUP_EPOCHS=25                   # Natural training epochs before adversarial training

# ATAS specific parameters
C=0.01                             # Hard fraction for adaptive step size
MAX_STEP_SIZE=12                   # Maximum perturbation step size
MIN_STEP_SIZE=4                    # Minimum perturbation step size
EPOCHS_RESET=6                     # Reset perturbations

# Output directories & resources
MODEL_DIR="./results/cifar_atas_resnet4b_ultrawide"
mkdir -p $MODEL_DIR
LOG_DIR="./log"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/cifar_atas_resnet4b_ultrawide.log"
NUM_WORKERS=4                      # Workers for data loading

# Display configuration
echo "==== CIFAR Adversarial Training (Ultra-Wide ResNet4b) ====="
echo "Dataset: $DATASET"
echo "Model: $ARCH (in_planes=64)"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Epsilon: $EPSILON/255"
echo "Results will be saved to: $MODEL_DIR"
echo "Log file: $LOG_FILE"
echo "=================================================="

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

# Evaluate at multiple perturbation sizes
for eps in 1 2 4 8; do
  ATTACK_LOG="${LOG_DIR}/attack_${DATASET}_${ARCH}_eps${eps}.log"
  echo "Evaluating at epsilon=${eps}/255"
  python3 -u attack.py \
    --dataset $DATASET \
    --model-dir $MODEL_DIR \
    --arch $ARCH \
    --epsilon $eps > $ATTACK_LOG 2>&1
done

echo "Attack evaluation completed! Results saved to log directory"

echo ""
echo "To prune this model:"
echo "python structured_prune_resnet4b.py --model-dir ./results/cifar_atas_resnet4b_ultrawide --output-dir ./results/cifar_atas_resnet4b_ultrawide_pruned --prune-iterations 10 --clean-epochs 5" 
