# VERL Framework Reasoning Reward Model Training Guide

This directory contains training scripts for fine-tuning a Reasoning Reward Model based on the VERL framework. The model is designed to evaluate and score responses based on their reasoning quality, supporting both single-node and multi-node distributed training.

[sft_rm.sh](../../../examples/train/sft_rm/sft_rm.sh)

## Environment Setup

Ensure the following dependencies are installed:
- VERL framework

## Data Processing

### Data Structure

The training script expects data in `.parquet` format with the following structure. Each sample represents a pair of responses with reasoning analysis and preference judgment:

```json
{
    "id": "unique-sample-id",
    "messages": [
        {
            "role": "user",
            "content": "# Task Description\nPlease choose the better response.\n\n# Principles\n1. Relevance and Specificity\n2. Depth of Analysis\n3. Clarity and Structure\n4. Practicality and Actionability\n5. User-Centric Focus\n\n# Query\n[Previous conversation history]\n\n# Response A\n[First response option]\n\n# Response B\n[Second response option]\n\n# Output Requirement\n[Output format requirements]"
        },
        {
            "role": "assistant",
            "content": "<think>\n[Detailed analysis of responses based on principles]\n</think>\n\n<preference>B</preference>"
        }
    ]
}
```

#### Required Fields

1. **Root Level**
   - `id`: Unique identifier for the training sample
   - `messages`: Array of conversation messages

2. **Messages**
   - Each message must have:
     - `role`: Either "user" or "assistant"
     - `content`: The message content

3. **User Message Structure**
   - Task Description: Clear instruction for comparing responses
   - Principles: Evaluation criteria for reasoning quality
   - Query: Original conversation context
   - Response options (A and B): Candidate responses to be compared
   - Output requirements: Format specifications for the analysis

4. **Assistant Message Structure**
   - `<think>` tag: Contains detailed reasoning analysis based on principles
   - `<preference>` tag: Final preference choice ("A", "B", or "tie")

### Example Data

Here's a simplified example showing how to structure the reasoning and preference data:

```json
{
    "id": "4def5dd5-7942-466d-a9e6-e651f348f81f",
    "messages": [
        {
            "role": "user",
            "content": "# Task Description\nPlease choose the better response.\n\n# Principles\n1. Relevance and Specificity\n2. Depth of Analysis\n3. Clarity and Structure\n4. Practicality and Actionability\n5. User-Centric Focus\n\n# Query\nuser: Rephrase 'the future of robotics and robots'\n\n# Response A\nRobotics and robots' future.\n\n# Response B\nRobotics and robots: tomorrow's advancements."
        },
        {
            "role": "assistant",
            "content": "<think>\nAnalyzing based on principles:\n1. Both responses are relevant\n2. Response B provides more context\n3. Response B has better structure with colon\n4. Both are practical and concise\n5. Response B better engages the user\n</think>\n\n<preference>B</preference>"
        }
    ]
}
```

## Training Configuration

### Single-Node Training

```bash
# Set environment variables
export MODEL_PATH="/path/to/your/model"
export TRAIN_FILE="/path/to/train.parquet"
export VAL_FILE="/path/to/val.parquet"
export SAVE_PATH="/path/to/checkpoints"
export PROJECT_NAME="reasoning-rm-experiment"

# Run training
bash sft_rm.sh
```

### Multi-Node Training

Set the same environment variables on each machine, plus distributed training parameters:

**Master Node (rank 0):**
```bash
export MASTER_ADDR="192.168.1.100"  # Master node IP
export MASTER_PORT="29500"          # Communication port
export WORLD_SIZE="2"               # Total number of nodes
export RANK="0"                     # Current node rank
export N_GPUS_PER_NODE="8"         # GPUs per node

bash sft_rm.sh
```

**Worker Node (rank 1):**
```bash
export MASTER_ADDR="192.168.1.100"  # Master node IP
export MASTER_PORT="29500"          # Communication port
export WORLD_SIZE="2"               # Total number of nodes
export RANK="1"                     # Current node rank
export N_GPUS_PER_NODE="8"         # GPUs per node

bash sft_rm.sh
```