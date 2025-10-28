#!/bin/bash
set -x

# Configuration
RUBRICS_PATH="./rubric_structuring_results/ready_to_use_rubrics.json"
DATASET_PATH="./data/helpsteer3_preference_valid.jsonl"
MODEL="qwen3-32b"
MAX_SAMPLES=100
MAX_WORKERS=256
OUTPUT_DIR="./rubric_analysis_results"

# Optional source rubrics for comparison (set to path to enable)
SOURCE_RUBRICS=""  # e.g., "./rubric_generation_output/rubrics.json"

python analysis.py \
    --rubrics $RUBRICS_PATH \
    --dataset $DATASET_PATH \
    --model $MODEL \
    --max-samples $MAX_SAMPLES \
    --max-workers $MAX_WORKERS \
    --output $OUTPUT_DIR \
    ${SOURCE_RUBRICS:+--source-rubrics $SOURCE_RUBRICS}

