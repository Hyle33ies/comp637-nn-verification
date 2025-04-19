#!/bin/bash

mkdir -p ./results/gtsrb_atas
mkdir -p ./log

# Set parameters
BATCH_SIZE=64
TEST_BATCH_SIZE=128
EPOCHS=30
EPSILON=8
LR=0.01
DECAY_STEPS="24 28"
C=0.01
MAX_STEP_SIZE=14
MIN_STEP_SIZE=4
MODEL_DIR=./results/gtsrb_atas
NUM_WORKERS=6
WARMUP_EPOCHS=2
DROPOUT=0.1

# Run ATAS adversarial training on GTSRB
echo "Starting ATAS adversarial training for GTSRB..."
python atas_gtsrb.py \
  --batch-size ${BATCH_SIZE} \
  --test-batch-size ${TEST_BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --decay-steps ${DECAY_STEPS} \
  --epsilon ${EPSILON} \
  --dropout-rate ${DROPOUT}\
  --max-step-size ${MAX_STEP_SIZE} \
  --min-step-size ${MIN_STEP_SIZE} \
  --c ${C} \
  --model-dir ${MODEL_DIR} \
  --num-workers ${NUM_WORKERS} \
  --warmup-epochs ${WARMUP_EPOCHS} | tee log/atas_gtsrb.log

echo "ATAS training completed!" 
