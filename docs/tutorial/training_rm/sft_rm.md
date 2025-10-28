# VERL Framework Reasoning Reward Model Training Guide

This directory contains training scripts for fine-tuning a Reasoning Reward Model based on the VERL framework. The model is designed to evaluate and score responses based on their reasoning quality, supporting both single-node and multi-node distributed training.


## Training Approaches Overview

This guide covers two main approaches for reward model training:

1. **Pairwise Training**: Compares pairs of responses to determine relative preference (Response A vs Response B)
2. **Pointwise Training**: Evaluates individual responses independently with absolute scoring

Both approaches use the same underlying VERL framework but differ in data structure and evaluation methodology.

## Environment Setup

Ensure the following dependencies are installed:
- VERL framework

## Data Processing

### Training Data Structure

This training approach supports both pointwise and pairwise training paradigms, expecting data in `.parquet` format with the following structure.

#### Required Fields

**Common Structure (Both Approaches):**

1. **Root Level**
   - `id`: Unique identifier for the training sample
   - `messages`: Array of conversation messages

2. **Messages**
   - Each message must have:
     - `role`: Either "user" or "assistant"
     - `content`: The message content


**Pairwise-Specific Fields:**

3. **User Message Structure**
   - Task Description: Clear instruction for comparing responses
   - Rubrics: Evaluation criteria for reasoning quality
   - Query: Original conversation context
   - Response options (A and B): Candidate responses to be compared
   - Output requirements: Format specifications for the analysis

4. **Assistant Message Structure**
   - `<think>` tag: Contains detailed reasoning analysis based on rubrics
   - `<preference>` tag: Final preference choice ("A", "B", or "tie")

**Pointwise-Specific Fields:**

3. **User Message Structure**
   - Task Description: Clear instruction for evaluating response quality
   - Rubrics: Evaluation criteria for reasoning quality
   - Query: Original conversation context
   - Context: Additional background information (optional)
   - Answer: The single response to be evaluated
   - Output requirements: Format specifications for scoring

4. **Assistant Message Structure**
   - `<think>` tag: Contains detailed reasoning analysis based on rubrics
   - `<score>` tag: Numerical score (typically 0-4 for helpfulness evaluation)

### Example Data

#### Pairwise Training Example

Here's a simplified example showing how to structure the reasoning and preference data:

```json
{
    "id": "4def5dd5-7942-466d-a9e6-e651f348f81f",
    "messages": [
        {
            "role": "user",
            "content": "# Task Description\nPlease choose the better response.\n\n# Rubrics\n1. Relevance and Specificity\n2. Depth of Analysis\n3. Clarity and Structure\n4. Practicality and Actionability\n5. User-Centric Focus\n\n# Query\nuser: Rephrase 'the future of robotics and robots'\n\n# Response A\nRobotics and robots' future.\n\n# Response B\nRobotics and robots: tomorrow's advancements."
        },
        {
            "role": "assistant",
            "content": "<think>\nAnalyzing based on rubrics:\n1. Both responses are relevant\n2. Response B provides more context\n3. Response B has better structure with colon\n4. Both are practical and concise\n5. Response B better engages the user\n</think>\n\n<preference>B</preference>"
        }
    ]
}
```

#### Pointwise Training Example

The pointwise training approach evaluates individual responses independently using absolute scoring. The data structure focuses on single responses with quality scores:

```json
{
    "id": "unique-sample-id",
    "messages": [
        {
            "role": "user",
            "content": "# Task Description\nEvaluate the helpfulness of the response on a scale from 0 to 4.\n\n# Rubrics\n1. Relevance and Specificity\n2. Depth of Analysis  \n3. Clarity and Structure\n4. Practicality and Actionability\n5. User-Centric Focus\n\n# Query\nuser: Summarize the novel \"The Green Mile\" in 200 words.\n\n# Context\n[Additional context if available]\n\n# Answer\n\"The Green Mile\" is a novel by Stephen King, published in 1996. It is set in the 1930s and follows the story of Paul Edgecombe, a corrections officer at a prison in Louisiana. The novel explores themes of justice, morality, and the human condition through the story of John Coffey, a gentle giant on death row who possesses healing abilities.\n\n# Output Requirement\nProvide your evaluation with detailed reasoning and a numerical score."
        },
        {
            "role": "assistant",
            "content": "<think>\nEvaluating based on rubrics:\n1. Relevance: High - directly addresses the request for a summary\n2. Depth: Good - covers main plot points and themes\n3. Clarity: Excellent - well-structured and easy to follow\n4. Practicality: Good - provides useful overview for someone unfamiliar with the book\n5. User-focus: Good - meets the specific word count and content requirements\n\nThe response provides a solid summary that captures key elements while staying within the requested length.\n</think>\n\n<score>3</score>"
        }
    ]
}
```

#### Key Differences: Pairwise vs Pointwise

| Aspect | Pairwise | Pointwise |
|--------|----------|-----------|
| **Evaluation Method** | Comparative (A vs B) | Absolute scoring |
| **Output Format** | `<preference>A/B/tie</preference>` | `<score>0-4</score>` |
| **Data Complexity** | Higher (requires 2 responses) | Lower (single response) |
| **Training Signal** | Relative ranking | Absolute quality scores |
| **Use Case** | Preference learning, RLHF | Quality assessment, filtering |

## Training Configuration

### Single-Node Training

```bash
# Set environment variables
export MODEL_PATH="/path/to/your/model"
export TRAIN_FILE="/path/to/train.parquet"
export VAL_FILE="/path/to/val.parquet"
export SAVE_PATH="/path/to/checkpoints"
export PROJECT_NAME="reasoning-sft-rm-experiment"

# Run training
bash sft_rm.sh

```

**Note**: The same training script `sft_rm.sh` can be used for both approaches. The framework automatically detects the data format based on the assistant message content (`<preference>` for pairwise, `<score>` for pointwise).

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