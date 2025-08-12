#!/bin/bash
set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# from environment variables
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${WORLD_SIZE:-2}
NODE_RANK=${RANK}

echo "=== Distributed Training Configuration ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "N_NODES: $NNODES"
echo "N_GPUS_PER_NODE: $N_GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"
echo "=========================================="

# configuration
SAVE_PATH=./checkpoints/sft
MODEL_PATH=Qwen/Qwen3-1.7B
TRAIN_FILE=./data/train.parquet
VAL_FILE=./data/test.parquet

PROJECT_NAME=rm-gallery-sft
EXPERIMENT_NAME=sft-${TIMESTAMP}

# use python -m torch.distributed.run for multi-node multi-gpu training
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=96 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key="null" \
    data.micro_batch_size=12 \
    data.truncation=right \
    model.enable_gradient_checkpointing=true \
    model.partial_pretrain=$MODEL_PATH \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.model_dtype="bf16" \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$PROJECT_NAME \
    data.max_length=8192 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','swanlab'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true
