#!/usr/bin/env bash

# Create output directory for pruned models
PRUNED_DIR="./results/cifar_atas_resnet18_pruned"
mkdir -p $PRUNED_DIR

# --- Configuration --- 
# Basic settings
DATASET="cifar10"                  # Dataset
ARCH="ResNet18"                    # Model architecture
MODEL_DIR="./results/cifar_atas_resnet18"  # Trained model directory

# Pruning settings
PRUNE_PERCENT=10.0                 # Percentage to prune in each iteration
PRUNE_ITERATIONS=9                 # Number of iterations to reach 90% sparsity
EXTRA_ITERATIONS="95.0 98.0"       # Additional sparsity levels to try

# Fine-tuning settings
FINETUNE_EPOCHS=5                  # Epochs for fine-tuning after each pruning
BATCH_SIZE=64                      # Batch size for training
TEST_BATCH_SIZE=128                # Batch size for testing
LR=0.01                            # Learning rate for fine-tuning (lower than original)
WEIGHT_DECAY=5e-4                  # Weight decay
MOMENTUM=0.9                       # SGD momentum

# Adversarial training parameters
EPSILON_RAW=8.0                      # Perturbation size (numerator)
STEP_SIZE_RAW=2.0                    # Step size for PGD (numerator)
PGD_STEPS=10                       # Number of PGD steps for training
PGD_EVAL_STEPS=20                  # Number of PGD steps for final evaluation

# Calculate float values for epsilon and step size
EPSILON=$(echo "$EPSILON_RAW / 255.0" | bc -l)
STEP_SIZE=$(echo "$STEP_SIZE_RAW / 255.0" | bc -l)

# Resources
NUM_WORKERS=4                      # Number of data loading workers
LOG_FILE="./log/prune_resnet18.log"
mkdir -p ./log

# Display configuration
echo "==== CIFAR ResNet18 Pruning Configuration ====="
echo "Dataset: $DATASET"
echo "Model: $ARCH"
echo "Trained model directory: $MODEL_DIR"
echo "Output directory: $PRUNED_DIR"
echo "Pruning iterations: $PRUNE_ITERATIONS @ $PRUNE_PERCENT% each"
echo "Extra sparsity levels: $EXTRA_ITERATIONS"
echo "Fine-tuning: $FINETUNE_EPOCHS epochs @ LR=$LR"
echo "Adversarial training: ε=$EPSILON_RAW/255 ($EPSILON), PGD-$PGD_STEPS steps"
echo "=============================================="

# Run pruning process
python3 -u prune_resnet18.py \
  --dataset $DATASET \
  --arch $ARCH \
  --model-dir "$MODEL_DIR" \
  --output-dir "$PRUNED_DIR" \
  --batch-size $BATCH_SIZE \
  --test-batch-size $TEST_BATCH_SIZE \
  --finetune-epochs $FINETUNE_EPOCHS \
  --lr $LR \
  --momentum $MOMENTUM \
  --weight-decay $WEIGHT_DECAY \
  --prune-percent $PRUNE_PERCENT \
  --prune-iterations $PRUNE_ITERATIONS \
  --extra-iterations "$EXTRA_ITERATIONS" \
  --epsilon "$EPSILON" \
  --step-size "$STEP_SIZE" \
  --pgd-steps $PGD_STEPS \
  --pgd-eval-steps $PGD_EVAL_STEPS \
  --num-workers $NUM_WORKERS \
  | tee $LOG_FILE

echo "Pruning process completed!"
echo "Results saved to $PRUNED_DIR"
echo "Log file: $LOG_FILE"

# Final model conversion for verification (optional)
echo "Converting the pruned model for verification..."

for SPARSITY in 90.0 95.0 98.0; do
  MODEL_PATH="$PRUNED_DIR/pruned_sparsity_$SPARSITY.pth"
  VERIFY_DIR="../alpha-beta-CROWN/complete_verifier/models/cifar10_resnet18_$SPARSITY"
  
  if [ -f "$MODEL_PATH" ]; then
    mkdir -p $VERIFY_DIR
    cp $MODEL_PATH $VERIFY_DIR/pruned.pth
    echo "Model with sparsity $SPARSITY% copied to $VERIFY_DIR"
  else
    echo "Warning: Model with sparsity $SPARSITY% not found at $MODEL_PATH"
  fi
done

echo "Process completed successfully!" 
