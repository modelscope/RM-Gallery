# Bradley-Terry Reward Model Training

This directory contains a complete implementation for training reward models using the Bradley-Terry preference learning approach. The Bradley-Terry model is widely used in RLHF (Reinforcement Learning from Human Feedback) to learn human preferences from pairwise comparisons.

## Overview

The Bradley-Terry model learns to assign scalar rewards to text completions by training on preference data. Given a pair of responses (chosen vs rejected) to the same prompt, the model learns to assign higher rewards to preferred responses.

This implementation is built on top of the **VERL (Versatile Efficient Reinforcement Learning)** framework and uses **FSDP (Fully Sharded Data Parallel)** for efficient distributed training of large reward models.

### Key Components

- **`trainer.py`**: FSDP-based Bradley-Terry trainer with VERL integration
- **`dataset.py`**: Base dataset processor for standard preference data formats
- **`dataset_helpsteer3.py`**: Custom dataset processor for complex nested formats like HelpSteer3
- **`evaluate.py`**: Simple reward model evaluation and comparison script
- **`run_bt.sh`**: Training script for standard preference datasets
- **`run_bt_helpsteer3.sh`**: Training script for HelpSteer3 format data
- **`trainer.yaml`**: Hydra configuration file with all training parameters

## Features

- **FSDP Training**: Fully Sharded Data Parallel for efficient large model training
- **VERL Integration**: Built on VERL framework for stable and efficient training
- **Bradley-Terry Loss**: Implements the standard BT loss: `-log(sigmoid(r_chosen - r_rejected))`
- **Custom Data Processors**: Flexible architecture for different dataset formats
- **Hydra Configuration**: Configuration management via Hydra with YAML files
- **Memory Optimization**: Gradient checkpointing, mixed precision (bf16), and CPU offloading
- **Advanced Schedulers**: Support for cosine and WSD learning rate schedules
- **Evaluation Metrics**: Accuracy computation based on preference ranking
- **Logging Integration**: Built-in support for console, WandB, and SwanLab logging

## Data Format & Preparation

### Dataset Format Options

The trainer supports two types of data formats:

#### 1. Standard Simple Format (BTDataset)

For simple preference datasets with direct text columns:

```python
# Parquet columns:
{
    'chosen': "Complete conversation text with preferred response",
    'rejected': "Complete conversation text with rejected response"
}
```

This format is processed by the default `BTDataset` class and works with datasets like `hendrydong/preference_700K`.

**Key Configuration Parameters:**
- `chosen_key`: Column name for preferred responses (default: "chosen")
- `rejected_key`: Column name for less preferred responses (default: "rejected")

You can customize these column names in your configuration:

```yaml
data:
  chosen_key: "winner"        # If your dataset uses "winner" column
  rejected_key: "loser"       # If your dataset uses "loser" column
```

**Data Processing Logic:**
1. The `BTDataset` reads parquet files and extracts text from specified columns
2. Each text string should contain the complete conversation (prompt + response)
3. The dataset tokenizes both chosen and rejected texts separately
4. Returns tokenized pairs: `input_ids_j` (chosen), `input_ids_k` (rejected)

#### 2. Complex Nested Format (Custom Datasets)

For complex preference datasets with nested structures, you can create custom dataset processors.

**Example: HelpSteer3 Format**

The included `HelpSteer3Dataset` handles this specific format:

[dataset_helpsteer3.py](../../../examples/train/bradley-terry/dataset_helpsteer3.py)

```json
{
  "input": [{"role": "user", "content": "Question"}],
  "output": [
    {
      "answer": {"role": "assistant", "content": "Response A"},
      "label": {"is_preferred": true}
    },
    {
      "answer": {"role": "assistant", "content": "Response B"},
      "label": {"is_preferred": false}
    }
  ]
}
```

### Creating Custom Dataset Processors

For datasets that don't match the simple chosen/rejected format, you need to create custom dataset processors.

#### Why Custom Datasets Are Needed

**Standard BTDataset Limitations:**
- Only works with simple text columns (chosen/rejected)
- Cannot handle nested JSON structures
- Cannot extract preferences from complex metadata
- Limited to predefined column names

**Custom Dataset Benefits:**
- Handle any data format (JSON, nested structures, etc.)
- Extract preferences from complex labeling schemes
- Apply custom preprocessing and filtering
- Support multiple conversation formats

#### Implementation Requirements

Create a custom dataset class that:

1. **Inherits from `torch.utils.data.Dataset`**
2. **Implements required methods:**
   - `__init__(parquet_files, tokenizer, config)`: Initialize with data loading
   - `__len__()`: Return dataset size
   - `__getitem__(idx)`: Return single sample
3. **Returns standardized format:**
   ```python
   {
       "input_ids_j": torch.Tensor,      # Chosen response tokens
       "attention_mask_j": torch.Tensor,  # Chosen attention mask
       "input_ids_k": torch.Tensor,      # Rejected response tokens
       "attention_mask_k": torch.Tensor   # Rejected attention mask
   }
   ```

#### Configuration and Loading

The trainer automatically loads custom dataset classes:

```yaml
data:
  custom_cls:
    path: ./dataset_helpsteer3.py  # Path to Python file (relative to trainer.py)
    name: HelpSteer3Dataset        # Class name in the module
```

**Key Logic:**
1. Trainer checks if `custom_cls.path` is specified
2. If yes, dynamically imports the module and loads the class
3. If no, uses default `BTDataset` with `chosen_key`/`rejected_key`

#### Custom Dataset Implementation Example

For a complete example, please refer to: [HelpSteer3Dataset](../../../examples/train/bradley-terry/dataset_helpsteer3.py)

#### Data Processing Pipeline

**For Standard BTDataset:**
```
Parquet File → Extract chosen_key/rejected_key columns → Tokenize → Return pairs
```

**For Custom Datasets:**
```
Parquet File → Custom parsing logic → Convert to chosen/rejected → Tokenize → Return pairs
```

Both approaches ultimately produce the same standardized output format that the trainer expects.

## Quick Start

### 1. Standard Preference Dataset Training

For datasets like `hendrydong/preference_700K` with simple chosen/rejected columns:

```bash
# cd into directory
cd ./examples/train/bradley-terry

# Make the script executable
chmod +x run_bt.sh

# Run training
./run_bt.sh
```

### 2. HelpSteer3 Format Training

For complex nested datasets like HelpSteer3:

```bash
# cd into directory
cd ./examples/train/bradley-terry

# Make the script executable
chmod +x run_bt_helpsteer3.sh

# Run training with custom dataset
./run_bt_helpsteer3.sh
```

### 3. Custom Configuration

You can modify the training parameters in several ways:

#### Option A: Edit the shell script
Modify variables in `run_bt.sh` or `run_bt_helpsteer3.sh`:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
TRAIN_FILE=hendrydong/preference_700K
VAL_FILE=null
```

#### Option B: Edit the YAML config
Modify `trainer.yaml` for persistent configuration changes.

#### Option C: Override via command line
Override specific parameters when running:

```bash
python -m torch.distributed.run \
    --nproc_per_node=8 \
    ./trainer.py \
    data.train_files=your_dataset \
    model.partial_pretrain=your_model \
    optim.lr=5e-7 \
    trainer.total_epochs=3
```

## Multi-Node Distributed Training

For training large reward models across multiple nodes, the trainer supports distributed training via `torch.distributed.run`.

### Single-Node Multi-GPU Training

For training on a single node with multiple GPUs:

```bash
# 8 GPUs on single node
python -m torch.distributed.run \
    --nproc_per_node=8 \
    ./trainer.py \
    [config_overrides]
```

### Multi-Node Training Setup

#### Prerequisites

1. **Network Configuration**: Ensure all nodes can communicate with each other
2. **Shared Storage**: All nodes should have access to the same dataset and model files
3. **Environment**: Same Python environment and dependencies on all nodes

#### Environment Variables

Set the following environment variables on all nodes:

```bash
# Master node configuration
export MASTER_ADDR="192.168.1.100"  # IP of the master node
export MASTER_PORT="29500"           # Port for distributed communication
export WORLD_SIZE=16                 # Total number of GPUs across all nodes
export NNODES=2                      # Total number of nodes

# Node-specific configuration (different for each node)
export RANK=0                        # Node rank: 0 for master, 1,2,... for workers
export N_GPUS_PER_NODE=8            # Number of GPUs per node
```

#### Execution Commands

**On Master Node (RANK=0):**
```bash
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"
export WORLD_SIZE=16
export NNODES=2
export RANK=0
export N_GPUS_PER_NODE=8

python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./trainer.py \
    data.train_batch_size=512 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    [other_config_overrides]
```

**On Worker Node (RANK=1):**
```bash
export MASTER_ADDR="192.168.1.100"  # Same master IP
export MASTER_PORT="29500"           # Same port
export WORLD_SIZE=16                 # Same total GPUs
export NNODES=2                      # Same node count
export RANK=1                        # Different rank for this node
export N_GPUS_PER_NODE=8            # Same GPUs per node

python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./trainer.py \
    data.train_batch_size=512 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    [other_config_overrides]
```

#### Using Training Scripts for Multi-Node

You can also modify the training scripts (`run_bt.sh` or `run_bt_helpsteer3.sh`) for multi-node training:

```bash
#!/bin/bash
# Multi-node configuration
export MASTER_ADDR=${MASTER_ADDR:-"192.168.1.100"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export WORLD_SIZE=${WORLD_SIZE:-16}    # 2 nodes × 8 GPUs
export NNODES=${NNODES:-2}
export RANK=${RANK:-0}                 # Set to 0,1,2... for each node
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# Run the same script on each node with appropriate RANK
./run_bt.sh
```

## Configuration Options

The trainer uses Hydra for configuration management. All parameters are defined in [trainer.yaml](../../../examples/train/bradley-terry/trainer.yaml) and can be overridden via command line.

### Data Configuration Deep Dive

#### chosen_key and rejected_key

These parameters are **only used by the default BTDataset** and are ignored when using custom datasets:

```yaml
data:
  chosen_key: chosen     # Column name for preferred responses
  rejected_key: rejected # Column name for less preferred responses
```

**When to modify:**
- Your parquet files use different column names (e.g., "winner"/"loser", "good"/"bad")
- You want to swap preference direction (though not recommended)

**Examples:**
```yaml
# For datasets with different column names
data:
  chosen_key: "response_1"
  rejected_key: "response_2"

# For datasets with winner/loser format
data:
  chosen_key: "winner_response"
  rejected_key: "loser_response"
```

#### custom_cls Configuration

For complex data formats that cannot be handled by simple column extraction:

```yaml
data:
  custom_cls:
    path: ./dataset_helpsteer3.py  # Python file path (relative to trainer.py)
    name: HelpSteer3Dataset        # Class name in the module
```

**Key Behavior:**
- If `custom_cls.path` is **not null**: Uses custom dataset class, ignores `chosen_key`/`rejected_key`
- If `custom_cls.path` is **null**: Uses default BTDataset with `chosen_key`/`rejected_key`

**Real Examples:**
```yaml
# Using HelpSteer3 custom dataset
data:
  custom_cls:
    path: ./dataset_helpsteer3.py
    name: HelpSteer3Dataset
  chosen_key: chosen    # These are ignored when custom_cls is used
  rejected_key: rejected

# Using default BTDataset
data:
  custom_cls:
    path: null           # Use default dataset
    name: null
  chosen_key: preferred  # These are used
  rejected_key: rejected
```


## Training Process

### Architecture

The training process is built on the **VERL framework** and uses **FSDP (Fully Sharded Data Parallel)** for efficient distributed training:

1. **FSDP Integration**: Model parameters are sharded across GPUs for memory efficiency
2. **VERL Framework**: Provides stable training utilities and optimizations
3. **Hydra Configuration**: All parameters managed through YAML configuration files
4. **Mixed Precision**: Uses bfloat16 for forward pass and float32 for gradient computation
5. **Gradient Checkpointing**: Trades compute for memory by recomputing activations during backward pass

### Loss Function
The Bradley-Terry loss is computed as:
```
L = -log(σ(r_chosen - r_rejected))
```
where `σ` is the sigmoid function and `r_chosen`, `r_rejected` are the reward scores.

### Data Flow
1. **Load Data**: Raw preference data loaded from parquet files via custom or default datasets
2. **Dataset Processing**: Custom dataset classes handle format conversion and tokenization
3. **FSDP Batching**: Chosen and rejected responses batched together with FSDP-aware collation
4. **Forward Pass**: Model computes reward scores for all responses using AutoModelForSequenceClassification
5. **Loss Computation**: Bradley-Terry loss computed on preference pairs with accuracy metrics
6. **FSDP Backward**: Gradients computed and synchronized across FSDP shards
7. **Optimization**: AdamW optimizer with cosine/WSD learning rate scheduling

### Evaluation

The [evaluate.py](../../../examples/train/bradley-terry/evaluate.py)provides a simple interface for testing trained Bradley-Terry reward models. It uses the `SimpleRewardEvaluator` class for easy reward scoring and response comparison.


#### Running Evaluation

```bash
python evaluate.py --model_path /path/to/trained/reward/model
```

#### Implementation Details

The `SimpleRewardEvaluator` provides:
- Easy-to-use interface for reward model testing
- Built-in conversation formatting using chat templates
- Automatic model loading with bfloat16 precision
- Direct reward score extraction from sequence classification outputs
- Response comparison utilities for preference ranking

## Additional Resources

### Training Scripts
- **[run_bt.sh](../../../examples/train/bradley-terry/run_bt.sh)**: Standard preference dataset training script
- **[run_bt_helpsteer3.sh](../../../examples/train/bradley-terry/run_bt_helpsteer3.sh)**: HelpSteer3 format training script
- **[trainer.yaml](../../../examples/train/bradley-terry/trainer.yaml)**: Complete configuration file with all parameters

### Dataset Processors
- **[dataset.py](../../../examples/train/bradley-terry/dataset.py)**: Default BTDataset for simple preference format
- **[dataset_helpsteer3.py](../../../examples/train/bradley-terry/dataset_helpsteer3.py)**: Custom processor for complex nested formats

### Core Implementation
- **[trainer.py](../../../examples/train/bradley-terry/trainer.py)**: FSDP Bradley-Terry trainer with VERL integration
- **[evaluate.py](../../../examples/train/bradley-terry/evaluate.py)**: Simple reward model evaluation utilities
