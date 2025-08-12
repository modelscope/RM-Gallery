#!/bin/bash
set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# Set environment variables for distributed training
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

echo "=== Distributed Training Configuration ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "N_NODES: $NNODES"
echo "N_GPUS_PER_NODE: $N_GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"
echo "=========================================="

# configuration
SAVE_PATH=./checkpoints/bt
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
TRAIN_FILE=hendrydong/preference_700K/train.parquet
VAL_FILE=hendrydong/preference_700K/test.parquet

PROJECT_NAME=rm-gallery-bt
EXPERIMENT_NAME=qwen2.5-7b-bt-${TIMESTAMP}

# Run with torchrun for multi-GPU FSDP trainings
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./trainer.py \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    model.partial_pretrain=$MODEL_PATH \
    data.max_length=4096 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=256 \
    optim.lr=5e-7 \
    optim.clip_grad=2 \
    trainer.total_epochs=3 \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    trainer.logger=['console','swanlab'] \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$SAVE_PATH/$EXPERIMENT_NAME
