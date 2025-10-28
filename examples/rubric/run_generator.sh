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
DOMAINS="general"  # Set to empty string "" to process all domains, or "multilingual" for specific domain
BATCH_SIZE=500

# Checkpoint and resume settings (uncomment to enable)
RESUME=""  # Set to "--resume" to enable
DISABLE_CHECKPOINT=""  # Set to "--disable-checkpoint" to disable

python generator.py \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --model $MODEL \
    --generate-number $GENERATE_NUMBER \
    --max-epochs $MAX_EPOCHS \
    --max-workers $MAX_WORKERS \
    --max-retries $MAX_RETRIES \
    --max-samples $MAX_SAMPLES \
    --batch-size $BATCH_SIZE \
    ${DOMAINS:+--domains $DOMAINS} \
    $RESUME \
    $DISABLE_CHECKPOINT


