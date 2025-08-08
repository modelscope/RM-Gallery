# Bradley-Terry Reward Model Training

This directory contains a complete implementation for training reward models using the Bradley-Terry preference learning approach. The Bradley-Terry model is widely used in RLHF (Reinforcement Learning from Human Feedback) to learn human preferences from pairwise comparisons.

## Overview

The Bradley-Terry model learns to assign scalar rewards to text completions by training on preference data. Given a pair of responses (chosen vs rejected) to the same prompt, the model learns to assign higher rewards to preferred responses.

### Key Components

- **`trainer.py`**: Main training script with Bradley-Terry loss implementation
- **`dataset.py`**: Data processor for converting preference data to training format
- **`run_qwen.sh`**: Example training script for Qwen model

## Features

- **Bradley-Terry Loss**: Implements the standard BT loss: `-log(sigmoid(r_chosen - r_rejected))`
- **Custom Data Processors**: Flexible architecture for different dataset formats
- **Distributed Training**: Support for multi-GPU training via Accelerate
- **Memory Optimization**: Gradient checkpointing and mixed precision (bf16)
- **Flexible Configuration**: Comprehensive argument parsing for all training parameters
- **Evaluation Metrics**: Accuracy computation based on preference ranking
- **Logging Integration**: Built-in support for WandB and SwanLab

## Data Format & Preparation

### Standard Internal Format

All data must be converted to the following **standard preference format** before training.

```python
{
    'chosen': [
        {"role": "user", "content": "Your question"},
        {"role": "assistant", "content": "Preferred response"}
    ],
    'rejected': [
        {"role": "user", "content": "Your question"},
        {"role": "assistant", "content": "Less preferred response"}
    ]
}
```

This is the **only format** the trainer accepts. Any data source with a different format must implement a custom dataset processor.

### Custom Dataset Implementation

**For any data source that doesn't match the standard format**, you must create a custom dataset processor by inheriting from `BaseBradleyTerryTrainDataset`.

### Example: HelpSteer3 Format

The included `HelpSteer3DataProcessor` shows how to handle this specific format.

[dataset.py](../../../examples/train/bradley-terry/dataset.py)

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

## Quick Start

### 1. Run Training

```bash
# cd into directory
cd ./examples/train/bradley-terry

# Make the script executable
chmod +x run_qwen_rm.sh

# Run training
./run_qwen_rm.sh
```

### 2. Custom Training Configuration

For custom training, modify the parameters in the training script or run directly:

```bash
accelerate launch trainer.py \
    --model_name "Qwen/Qwen3-1.7B" \
    --train_set_path "./data/train.parquet" \
    --eval_set_path "./data/eval.parquet" \
    --output_path "./models/reward_model" \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64
```

## Configuration Options

### Model & Training
- `--model_name`: Base model to fine-tune (default: Qwen/Qwen3-1.7B)
- `--max_length`: Maximum sequence length (default: 4096)
- `--learning_rate`: Learning rate (default: 2e-6)
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--bf16`: Use bfloat16 mixed precision (default: True)

### Data Processing
- `--train_set_path`: Path to training data file
- `--eval_set_path`: Path to evaluation data file
- `--custom_bt_dataset_path`: Path to custom dataset processor
- `--custom_bt_dataset_name`: Name of the dataset processor class

### Performance & Memory
- `--per_device_train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 64)
- `--gradient_checkpointing`: Enable gradient checkpointing (default: True)
- `--deepspeed`: Path to DeepSpeed config for large models

### Monitoring
- `--report_to`: Logging service ("wandb" or "swanlab")
- `--run_name`: Name for the training run
- `--logging_steps`: Steps between logging updates

## Training Process

### Loss Function
The Bradley-Terry loss is computed as:
```
L = -log(σ(r_chosen - r_rejected))
```
where `σ` is the sigmoid function and `r_chosen`, `r_rejected` are the reward scores.

### Data Flow
1. **Load Data**: Raw preference data loaded from parquet files
2. **Convert Format**: Data converted to chosen/rejected conversation pairs
3. **Tokenization**: Conversations tokenized using model's chat template
4. **Batching**: Chosen and rejected responses batched together
5. **Forward Pass**: Model computes reward scores for all responses
6. **Loss Computation**: Bradley-Terry loss computed on preference pairs

### Evaluation

The evaluation script computes and displays reward scores for pairs of responses, helping analyze the model's preference assignments.

[evaluate.py](../../../examples/train/bradley-terry/evaluate.py)

#### Running Evaluation

```bash
python evaluate.py \
    --model_path /path/to/model \
    --data_path /path/to/test.parquet \
    --max_length 8192
```

The scores represent the model's reward values for each response, with higher scores indicating stronger model preference.

#### Implementation Details

The `evaluate.py` script provides:
- Direct reward score computation for response pairs
- Clear output format with question context
- Robust error handling for long sequences
- Support for different model architectures and tokenizers

