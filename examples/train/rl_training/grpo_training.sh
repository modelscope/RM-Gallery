#!/bin/bash
# GRPO Training with RM-Gallery LLM Judge
# Supports multiple evaluation modes (winrate/copeland/dgr/elo/pointwise/listwise)

TIMESTAMP=$(date "+%m%dT%H%M")

# ============================================================================
# Judge Model Configuration (RM-Gallery)
# ============================================================================
export JUDGE_MODEL_NAME="qwen3-32b"
export JUDGE_API_URL="http://your-api-url/v1/chat/completions"
export JUDGE_API_KEY="your-api-key"

# Evaluation Mode
export EVAL_MODE="pairwise"        # pairwise, pointwise, listwise
export PAIRWISE_MODE="winrate"         # dgr, copeland, winrate, elo (only for pairwise)
export MAX_WORKERS=10
export VERBOSE="false"

# ============================================================================
# Performance & Timeout Configuration (prevent blocking)
# ============================================================================
export LLM_TIMEOUT="30.0"          # LLM call timeout per request (seconds)
export MAX_RETRIES="2"             # LLM retry attempts on failure

# ============================================================================
# Path Configuration
# ============================================================================

# VERL Root (change this!)
BASE_DIR="/path/to/verl"

# Model Path
MODEL_PATH="/path/to/base/model"

# Data Paths (absolute paths)
TRAIN_FILE="/path/to/rm-gallery/examples/train/rl_training/data/wildchat_10k_train.parquet"
VAL_FILE="/path/to/rm-gallery/examples/train/rl_training/data/wildchat_10k_test.parquet"

# Custom Module Paths (absolute paths to RM-Gallery examples)
RM_GALLERY_EXAMPLE_DIR="/path/to/rm-gallery/examples/train/rl_training"
CUSTOM_REWARD_FUNCTION_PATH="${RM_GALLERY_EXAMPLE_DIR}/alignment_reward_fn.py"
CUSTOM_CHAT_RL_DATASET_PATH="${RM_GALLERY_EXAMPLE_DIR}/alignment_rl_dataset.py"

# Project Configuration
PROJECT_NAME=alignment_grpo
EXPERIMENT_NAME=grpo_dgr_rm_gallery

# Hardware
N_GPUS_PER_NODE=8
N_NODES=1

# ============================================================================
# Optional: Performance Optimization
# ============================================================================

# Uncomment for better performance on InfiniBand clusters
# export NCCL_IBEXT_DISABLE=1
# export NCCL_NVLS_ENABLE=1
# export NCCL_IB_HCA=mlx5

# Environment variables for optimization
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0

set -x

# ============================================================================
# Run Training
# ============================================================================

cd "${BASE_DIR}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env="${BASE_DIR}/verl/trainer/runtime_env.yaml" \
    -- \
    python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=96 \
    data.val_batch_size=192 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.prompt_key='x' \
    data.custom_cls.path="${CUSTOM_CHAT_RL_DATASET_PATH}" \
    data.custom_cls.name="AlignmentChatRLDataset" \
    reward_model.reward_manager='dgr' \
    reward_model.launch_reward_fn_async=True \
    custom_reward_function.path="${CUSTOM_REWARD_FUNCTION_PATH}" \
    custom_reward_function.name='compute_score' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=10 \
    trainer.test_freq=2 \
    trainer.total_epochs=25 \
    trainer.val_before_train=True \
    trainer.default_local_dir="${BASE_DIR}/checkpoints/${EXPERIMENT_NAME}" $@

echo "Training completed! Checkpoints saved to: ${BASE_DIR}/checkpoints/${EXPERIMENT_NAME}"

