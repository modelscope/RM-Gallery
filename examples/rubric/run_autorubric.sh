#!/bin/bash
set -x

# Data and Model
DATA_PATH="./data/helpsteer3_preference_train.jsonl"
MODEL="qwen3-32b"
OUTPUT_BASE="./exports"

# Performance Settings
MAX_WORKERS=32
BATCH_SIZE=10
MAX_EPOCHS=10
GENERATE_NUMBER=1

# MCR Settings
MCR_BATCH_SIZE=10
MIN_INCREMENT_THRESHOLD=0.002
PATIENCE=2
MAX_ITERATIONS=50
MAX_TOTAL_RUBRICS=200
MIN_SUCCESS_RATE=0.3

# Structuring Settings
NUM_CATEGORIES=5
ENABLE_STRUCTURING=true

python auto_rubric.py \
    --data-path $DATA_PATH \
    --model $MODEL \
    --output-base $OUTPUT_BASE \
    --max-workers $MAX_WORKERS \
    --batch-size $BATCH_SIZE \
    --max-epochs $MAX_EPOCHS \
    --generate-number $GENERATE_NUMBER \
    --mcr-batch-size $MCR_BATCH_SIZE \
    --min-increment-threshold $MIN_INCREMENT_THRESHOLD \
    --patience $PATIENCE \
    --max-iterations $MAX_ITERATIONS \
    --max-total-rubrics $MAX_TOTAL_RUBRICS \
    --min-success-rate $MIN_SUCCESS_RATE \
    --enable-structuring $ENABLE_STRUCTURING \
    --num-categories $NUM_CATEGORIES


