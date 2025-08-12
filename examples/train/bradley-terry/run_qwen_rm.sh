#!/bin/bash

# Activate your Python environment if needed
# source /path/to/your/venv/bin/activate
TIMESTAMP=$(date "+%m%dT%H%M")

# Launch training with accelerate
accelerate launch ./trainer.py \
    --model_name Qwen/Qwen3-1.7B \
    --max_length 4096 \
    --custom_bt_dataset_path ./dataset.py \
    --custom_bt_dataset_name HelpSteer3DataProcessor \
    --train_set_path ./data/train.parquet \
    --eval_set_path ./data/test.parquet \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-5 \
    --bf16 true \
    --num_train_epochs 2 \
    --report_to "swanlab" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --output_path "./models/reward_model" \
    --gradient_checkpointing true \
    --run_name "qwen_1.7b_reward_model-${TIMESTAMP}"
