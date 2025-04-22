#!/bin/bash

# Check if bc is installed, install if not
if ! command -v bc &> /dev/null
then
    echo "bc could not be found, attempting to install..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y bc
    elif command -v yum &> /dev/null; then
        yum install -y bc
    else
        echo "Error: Cannot install bc. Please install it manually." >&2
        exit 1
    fi
    if ! command -v bc &> /dev/null; then
        echo "Error: Failed to install bc." >&2
        exit 1
    fi
fi

# Iterative Pruning and Fine-tuning Script for GTSRB WRN-28-4

# --- Configuration ---
INITIAL_MODEL="../Adv-train/results/gtsrb_atas_wrn28_4/best.pt" # Path to the initial robustly trained model
ARCH="wrn_28_4"                                                # Model architecture
NUM_CLASSES=43                                                 # Number of GTSRB classes
FINAL_SPARSITY=0.9                                             # Target final sparsity (e.g., 0.9 for 90% pruned)
PRUNE_STEP=0.1                                                 # Sparsity increase per iteration (e.g., 0.1 for 10%)
FINETUNE_EPOCHS=6                                             # Epochs for ATAS fine-tuning after each pruning step
FINETUNE_LR=0.001                                              # Learning rate for ATAS fine-tuning
FINETUNE_DECAY_STEPS="3 5"                                   # LR decay schedule for fine-tuning
FINETUNE_WARMUP=0                                              # Warmup epochs for fine-tuning (can be 0 or 1)
BASE_OUTPUT_DIR="./trained_models/gtsrb_iter_lwm_${ARCH}"      # Base directory to save results
# GPU_ID="0"

# --- Safety Checks ---
if [ ! -f "$INITIAL_MODEL" ]; then
    echo "Error: Initial model not found at $INITIAL_MODEL"
    exit 1
fi

# --- Setup ---
mkdir -p ${BASE_OUTPUT_DIR}/logs
current_model_path=$(realpath $INITIAL_MODEL) # Use absolute path for initial model
current_sparsity=0.0
num_steps=$(echo "scale=0; ($FINAL_SPARSITY / $PRUNE_STEP)" | bc)
ITER_PRUNE_DIR=$(pwd) # Save the current directory

echo "Starting Iterative Pruning up to ${FINAL_SPARSITY} sparsity..."
echo "Initial Model: ${INITIAL_MODEL}"
echo "Architecture: ${ARCH}"
echo "Pruning Step: ${PRUNE_STEP}"
echo "Fine-tuning Epochs per step: ${FINETUNE_EPOCHS}"
echo "Fine-tuning LR: ${FINETUNE_LR}"
echo "Total Iterations: ${num_steps}"

# --- Iterative Pruning Loop ---
for i in $(seq 1 $num_steps)
do
    # Calculate target sparsity for this iteration
    target_sparsity=$(echo "scale=2; $current_sparsity + $PRUNE_STEP" | bc)
    # Ensure target sparsity doesn't exceed final target
    target_sparsity=$(echo "if($target_sparsity > $FINAL_SPARSITY) $FINAL_SPARSITY else $target_sparsity" | bc)
    sparsity_pct=$(echo "scale=0; $target_sparsity * 100 / 1" | bc)

    iteration_dir_relative="iter_${i}_sparsity_${sparsity_pct}"
    iteration_dir="${BASE_OUTPUT_DIR}/${iteration_dir_relative}"
    mkdir -p ${iteration_dir}
    # Use absolute paths for prune script arguments
    pruned_model_path_abs="${ITER_PRUNE_DIR}/${iteration_dir}/pruned_lwm.pth.tar"
    finetune_log_path="${ITER_PRUNE_DIR}/${BASE_OUTPUT_DIR}/logs/finetune_iter_${i}.log"
    prune_log_path="${ITER_PRUNE_DIR}/${BASE_OUTPUT_DIR}/logs/prune_iter_${i}.log"

    echo "-----------------------------------------------------"
    echo "Iteration ${i}/${num_steps}: Target Sparsity ${target_sparsity} (${sparsity_pct}% pruned)"
    echo "-----------------------------------------------------"

    # 1. Prune the current model using LWM (Run from ITER_Prune directory)
    echo "[Iter ${i}] Pruning model from ${current_model_path} to ${target_sparsity} sparsity..."
    python prune_lwm.py \
        --source-net "${current_model_path}" \
        --target-sparsity ${target_sparsity} \
        --output-path "${pruned_model_path_abs}" \
        --arch ${ARCH} \
        --num-classes ${NUM_CLASSES} \
        | tee ${prune_log_path}

    if [ $? -ne 0 ]; then
        echo "Error during pruning step ${i}. Exiting."
        exit 1
    fi

    # 2. Fine-tune the pruned model using ATAS
    echo "[Iter ${i}] Fine-tuning pruned model ${pruned_model_path_abs} using ATAS..."

    # Record start time for fine-tuning
    FINETUNE_START_TIME=$(date +%s)
    
    # Change to Adv-train directory
    cd ../Adv-train

    # Adjust paths to be relative to Adv-train
    finetuned_model_output_dir_rel="../ITER_Prune/${iteration_dir}/finetuned_atas"
    pruned_model_input_path_rel="../ITER_Prune/${iteration_dir}/pruned_lwm.pth.tar"

    # Run ATAS fine-tuning
    python atas_gtsrb.py \
        --model-dir "${finetuned_model_output_dir_rel}" \
        --arch WideResNet `# ATAS script expects the base arch name` \
        --source-net "${pruned_model_input_path_rel}" `# Path to the LWM pruned model` \
        --lr ${FINETUNE_LR} \
        --epochs ${FINETUNE_EPOCHS} \
        --decay-steps ${FINETUNE_DECAY_STEPS} \
        --warmup-epochs ${FINETUNE_WARMUP} \
        --weight-decay 5e-4 `# Match ATAS default` \
        --momentum 0.9 `# Match ATAS default` \
        --epsilon 8 `# Epsilon / 255 is handled inside atas_gtsrb.py` \
        --max-step-size 14 \
        --min-step-size 4 \
        --c 0.01 \
        --epochs-reset 3 `# Adjust shorter for fine-tuning` \
        --batch-size 64 \
        --test-batch-size 128 \
        --num-workers 6 \
        --seed 42 `# Use consistent seed` \
        | tee ${finetune_log_path}

    finetune_exit_code=$?

    # Change back to ITER_Prune directory
    cd ${ITER_PRUNE_DIR}

    # Calculate fine-tuning duration
    FINETUNE_END_TIME=$(date +%s)
    FINETUNE_DURATION=$((FINETUNE_END_TIME - FINETUNE_START_TIME))
    HOURS=$((FINETUNE_DURATION / 3600))
    MINUTES=$(( (FINETUNE_DURATION % 3600) / 60 ))
    SECONDS=$((FINETUNE_DURATION % 60))
    
    # Print and append duration to the log file
    echo "[Iter ${i}] Fine-tuning completed in ${HOURS}h ${MINUTES}m ${SECONDS}s (${FINETUNE_DURATION} seconds total)"
    echo "[Iter ${i}] Fine-tuning completed in ${HOURS}h ${MINUTES}m ${SECONDS}s (${FINETUNE_DURATION} seconds total)" >> ${finetune_log_path}

    if [ $finetune_exit_code -ne 0 ]; then
        echo "Error during fine-tuning step ${i}. Exiting."
        exit 1
    fi

    # Update current model path (use absolute path) and sparsity for the next iteration
    current_model_path="${ITER_PRUNE_DIR}/${iteration_dir}/finetuned_atas/best.pt"
    current_sparsity=$target_sparsity

    # Check if the fine-tuned model exists before proceeding
    if [ ! -f "$current_model_path" ]; then
        echo "Error: Fine-tuned model best.pt not found at ${current_model_path}. Exiting."
        # Fallback to last.pt if best.pt is missing?
        current_model_path_last="${ITER_PRUNE_DIR}/${iteration_dir}/finetuned_atas/last.pt"
        if [ ! -f "$current_model_path_last" ]; then
           echo "Error: last.pt also not found. Exiting."
           exit 1
        else
           echo "Warning: best.pt not found, using last.pt instead for next iteration."
           current_model_path=$current_model_path_last
        fi
    fi

done

echo "-----------------------------------------------------"
echo "Iterative Pruning completed!"
echo "Final model at ${FINAL_SPARSITY} sparsity (approx): ${current_model_path}"
echo "Results saved in: ${BASE_OUTPUT_DIR}"
echo "Logs saved in: ${BASE_OUTPUT_DIR}/logs" 
