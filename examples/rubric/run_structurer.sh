#!/bin/bash
set -x

INPUT_FILE="./rubric_generation_output/rubrics.json"
OUTPUT_DIR="./rubric_structuring_results"
MODEL="qwen3-32b"
NUM_THEMES=5

python run_rubric_structurer.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL" \
    --themes $NUM_THEMES

