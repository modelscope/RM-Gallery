#!/bin/bash
set -x

DATA_PATH="./data/helpsteer3_preference_train.jsonl"
OUTPUT_DIR="./rubric_generation_output"
MODEL="qwen3-32b"
MAX_SAMPLES=200
GENERATE_NUMBER=1
MAX_EPOCHS=10
MAX_WORKERS=256
MAX_RETRIES=5
ENABLE_THINKING="true"
DOMAINS="multilingual"
BATCH_SIZE=500

# Checkpoint and resume settings (uncomment to enable)
# RESUME="--resume"
# DISABLE_CHECKPOINT="--disable-checkpoint"

python run_rubric_generator.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --generate-number $GENERATE_NUMBER \
    --max-epochs $MAX_EPOCHS \
    --max-workers $MAX_WORKERS \
    --max-retries $MAX_RETRIES \
    --enable-thinking $ENABLE_THINKING \
    --max-samples $MAX_SAMPLES \
    --domains "$DOMAINS" \
    --batch-size $BATCH_SIZE \
    $RESUME \
    $DISABLE_CHECKPOINT


