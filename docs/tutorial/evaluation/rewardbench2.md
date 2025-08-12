# RewardBench2 Evaluation Tutorial

## Overview

RewardBench2 is a comprehensive evaluation benchmark for reward models that tests their ability to rank multiple responses to a given query. This tutorial demonstrates how to use RM-Gallery's RewardBench2 evaluator to assess your reward models' performance.

The RewardBench2 evaluation protocol uses a list-wise comparison approach where the model selects the best response from multiple candidates, providing insights into the model's preference alignment and ranking capabilities.

## Features

- **List-wise Evaluation**: Compares multiple responses simultaneously rather than pairwise comparisons
- **Position Bias Mitigation**: Automatically shuffles responses to prevent position-based biases
- **Comprehensive Metrics**: Provides accuracy metrics overall and by subset categories
- **Parallel Processing**: Supports multi-threaded evaluation for faster processing

## Data Preparation

### Step 1: Download the Dataset

First, create the data directory and download the RewardBench2 dataset:

```bash
# Create the benchmark data directory
mkdir -p data/benchmarks

# Navigate to the directory
cd data/benchmarks

# Clone the RewardBench2 dataset from Hugging Face
git clone https://huggingface.co/datasets/allenai/reward-bench-2
```

### Step 2: Verify Data Structure

After downloading, your data structure should look like:

```
data/
└── benchmarks/
    └── reward-bench-2/
        ├── data/
        │   ├── test-00000-of-00001.parquet
        │   └── ...
        └── README.md
```

## Environment Setup

### Prerequisites

Ensure you have the required environment variables set up for your language model:

```bash
# For OpenAI-compatible APIs
export OPENAI_API_KEY="your_api_key_here"
export BASE_URL="your_base_url_here"  # Optional, for custom endpoints
```

**Environment Variables Check:**
```bash
# Verify environment variables are set
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:8}..."  # Shows first 8 characters
echo "BASE_URL: $BASE_URL"

# Or check if they exist
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY is not set"
else
    echo "✅ OPENAI_API_KEY is set"
fi
```

### Installation

Make sure RM-Gallery is installed:

```bash
pip install rm-gallery
```

### Quick Installation Check

Verify your installation by running the help command:

```bash
python rm_gallery/gallery/evaluation/rewardbench2.py --help
```

This should display the available command-line options. If you see an error about missing modules, ensure all dependencies are installed correctly.

## Basic Usage

### Quick Start

The easiest way to run RewardBench2 evaluation is directly from the command line:

```bash
# Simplest command with default parameters
python rm_gallery/gallery/evaluation/rewardbench2.py

# Or with custom parameters
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/rewardbench2_results.json" \
    --max_samples 100 \
    --model "deepseek-chat" \
    --max_workers 8
```

**Expected Output:**
```
Overall Accuracy: 0.7500
Valid Samples: 100
Model: deepseek-chat
Results saved to: data/results/rewardbench2_results.json
```

### Command Line Parameters

All parameters are optional and have default values:

```bash
# Minimal command with defaults
python rm_gallery/gallery/evaluation/rewardbench2.py

# Full command with all parameters
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/rewardbench2.json" \
    --max_samples 10 \
    --model "deepseek-chat" \
    --max_workers 8
```

### Programmatic Usage

You can also use the evaluation in your Python code:

```python
from rm_gallery.gallery.evaluation.rewardbench2 import main

# Run evaluation with custom settings
main(
    data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path="data/results/rewardbench2_results.json",
    max_samples=100,
    model="deepseek-chat",
    max_workers=8
)
```

### Important Notes

1. **Data Path**: Make sure the data path points to a valid RewardBench2 dataset file. The default path assumes you've downloaded the dataset to `data/benchmarks/reward-bench-2/`.

2. **Results Directory**: The script will create the results directory if it doesn't exist. Make sure you have write permissions.

3. **Model Configuration**: For simple model names, use the `--model` parameter. For complex configurations (temperature, max_tokens, etc.), use the programmatic approach.

4. **Environment Variables**: Ensure `OPENAI_API_KEY` and optionally `BASE_URL` are set before running the evaluation.

## Advanced Usage

### Custom Model Configuration

For simple model configurations, you can use the command line:

```bash
# Use a different model
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/rewardbench2_custom.json" \
    --max_samples 500 \
    --model "qwen3-32b" \
    --max_workers 16
```

For more complex model configurations, use the programmatic approach:

```python
from rm_gallery.gallery.evaluation.rewardbench2 import main

# Custom model configuration
model_config = {
    "model": "qwen3-32b",
    "temperature": 0.1,
    "max_tokens": 2048,
    "enable_thinking": True
}

main(
    data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path="data/results/rewardbench2_custom.json",
    max_samples=500,
    model=model_config,
    max_workers=16
)
```

### Programmatic Usage

For more control over the evaluation process:

```python
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.gallery.evaluation.rewardbench2 import RewardBench2Evaluator, RewardBench2Reward
from rm_gallery.core.utils.file import write_json

# 1. Set up data loading
config = {
    "path": "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    "limit": 100,  # Limit samples for testing
}

load_module = create_loader(
    name="rewardbench2",
    load_strategy_type="local",
    data_source="rewardbench2",
    config=config,
)

# 2. Initialize language model
llm = OpenaiLLM(model="deepseek-chat", enable_thinking=True)

# 3. Load dataset
dataset = load_module.run()

# 4. Create evaluator
evaluator = RewardBench2Evaluator(
    reward=RewardBench2Reward(
        name="rewardbench2",
        llm=llm,
        max_workers=8,
    )
)

# 5. Run evaluation
results = evaluator.run(samples=dataset.get_data_samples())

# 6. Save results
write_json(results, "data/results/rewardbench2_detailed.json")

# 7. Print summary
print(f"Overall Accuracy: {results['overall_accuracy']['accuracy']:.4f}")
print(f"Valid Samples: {results['overall_accuracy']['valid_samples']}")
```

## Configuration Parameters

### Main Function Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | Required | Path to the RewardBench2 dataset file |
| `result_path` | str | Required | Path to save evaluation results |
| `max_samples` | int | 10 | Maximum number of samples to evaluate |
| `model` | str/dict | "deepseek-chat" | Model name or configuration dictionary |
| `max_workers` | int | 8 | Number of parallel workers for evaluation |

### Model Configuration Options

When passing a model configuration dictionary:

```python
model_config = {
    "model": "qwen3-32b",           # Model name
    "temperature": 0.1,             # Sampling temperature
    "max_tokens": 2048,             # Maximum response tokens
    "enable_thinking": True,        # Enable thinking process
    "top_p": 0.9,                  # Top-p sampling
    "frequency_penalty": 0.0,       # Frequency penalty
    "presence_penalty": 0.0,        # Presence penalty
}
```

## Understanding Results

### Result Structure

The evaluation results contain:

```json
{
    "model": "deepseek-chat",
    "overall_accuracy": {
        "accuracy": 0.75,
        "correct_count": 75,
        "valid_samples": 100,
        "total_samples": 100,
        "choice_distribution": {
            "0": 25,
            "1": 30,
            "2": 25,
            "3": 20
        }
    },
    "subset_accuracy": {
        "chat": {
            "accuracy": 0.80,
            "correct_count": 40,
            "valid_samples": 50,
            "total_samples": 50
        },
        "reasoning": {
            "accuracy": 0.70,
            "correct_count": 35,
            "valid_samples": 50,
            "total_samples": 50
        }
    }
}
```

### Key Metrics

- **accuracy**: Proportion of correct predictions (0.0 to 1.0)
- **correct_count**: Number of correctly identified best responses
- **valid_samples**: Number of successfully processed samples
- **choice_distribution**: Distribution of selected best response positions

### Interpreting Results

1. **Overall Accuracy**: Higher values indicate better alignment with human preferences
2. **Subset Performance**: Compare performance across different task categories
3. **Choice Distribution**: Check for position bias - should be roughly uniform
4. **Valid Samples**: Ensure most samples were processed successfully

## Troubleshooting

### Common Issues

1. **Missing API Key**:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

2. **Data Path Issues**:
   - Ensure the parquet file exists at the specified path
   - Check file permissions

3. **Memory Issues**:
   - Reduce `max_samples` for large datasets
   - Adjust `max_workers` based on available resources

4. **Model Connection Issues**:
   - Verify `BASE_URL` is correct for custom endpoints
   - Check network connectivity

### Performance Optimization

1. **Parallel Processing**: Increase `max_workers` for faster evaluation
2. **Batch Size**: Process samples in smaller batches for memory efficiency
3. **Model Selection**: Use faster models for preliminary evaluation

## Best Practices

1. **Sample Size**: Start with small samples (10-100) for testing, then scale up
2. **Position Bias**: The evaluator automatically handles position bias through shuffling
3. **Result Validation**: Always check the `valid_samples` count in results
4. **Subset Analysis**: Analyze performance across different task categories
5. **Reproducibility**: Set random seeds for consistent results across runs

## Examples

### Example 1: Quick Evaluation

```bash
# Quick evaluation for testing
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/quick_test.json" \
    --max_samples 50 \
    --model "deepseek-chat" \
    --max_workers 4
```

### Example 2: Production Evaluation

```bash
# Full evaluation with optimized settings
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/production_eval.json" \
    --max_samples 1000 \
    --model "qwen3-32b" \
    --max_workers 16
```

For complex model configurations, use the programmatic approach:

```python
from rm_gallery.gallery.evaluation.rewardbench2 import main

# Full evaluation with custom model configuration
main(
    data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path="data/results/production_eval.json",
    max_samples=1000,
    model={
        "model": "qwen3-32b",
        "temperature": 0.1,
        "enable_thinking": True
    },
    max_workers=16
)
```

### Example 3: Multiple Model Comparison

```bash
# Compare different models
models=("deepseek-chat" "qwen3-32b" "gpt-4o-mini")

for model in "${models[@]}"; do
    echo "Evaluating with model: $model"
    python rm_gallery/gallery/evaluation/rewardbench2.py \
        --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
        --result_path "data/results/rewardbench2_${model//\//_}.json" \
        --max_samples 100 \
        --model "$model" \
        --max_workers 8
done

echo "All model evaluations completed!"
```

## Next Steps

After running RewardBench2 evaluation:

1. **Analyze Results**: Review accuracy metrics and subset performance
2. **Compare Models**: Run evaluations with different models for comparison
3. **Optimize Performance**: Use insights to improve your reward model
4. **Integration**: Integrate evaluation into your development pipeline

For more advanced evaluation scenarios, check out other evaluation tutorials in the RM-Gallery documentation.

## Integration with Other Benchmarks

RewardBench2 can be integrated with other evaluation frameworks:

```bash
# Example: Combining with JudgeBench
echo "Running RewardBench2 evaluation..."
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/rewardbench2.json" \
    --max_samples 100 \
    --model "deepseek-chat"

echo "Running JudgeBench evaluation..."
python rm_gallery/gallery/evaluation/judgebench.py \
    --data_path "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
    --result_path "data/results/judgebench.json" \
    --judge_type "arena_hard" \
    --max_samples 100 \
    --model "deepseek-chat"

echo "Both evaluations completed!"
```

Or using Python to run both evaluations:

```python
import subprocess
import sys

# Example: Combining with JudgeBench
def run_evaluation(script_path, **kwargs):
    """Helper function to run evaluation scripts"""
    cmd = [sys.executable, script_path]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {script_path} completed successfully")
    else:
        print(f"❌ {script_path} failed: {result.stderr}")
    return result.returncode == 0

# Run both evaluations
print("Running RewardBench2 evaluation...")
rewardbench2_success = run_evaluation(
    "rm_gallery/gallery/evaluation/rewardbench2.py",
    data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path="data/results/rewardbench2.json",
    max_samples=100,
    model="deepseek-chat"
)

print("Running JudgeBench evaluation...")
judgebench_success = run_evaluation(
    "rm_gallery/gallery/evaluation/judgebench.py",
    data_path="data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
    result_path="data/results/judgebench.json",
    judge_type="arena_hard",
    max_samples=100,
    model="deepseek-chat"
)

if rewardbench2_success and judgebench_success:
    print("🎉 Both evaluations completed successfully!")
else:
    print("⚠️  Some evaluations failed. Check the logs above.")
``` 