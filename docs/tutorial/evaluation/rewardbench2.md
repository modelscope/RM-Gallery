# RewardBench2 Evaluation Tutorial

## Overview

RewardBench2 is a comprehensive evaluation benchmark for reward models that tests their ability to rank multiple responses to a given query. This tutorial demonstrates how to use RM-Gallery's RewardBench2 evaluator to assess your reward models' performance.

The RewardBench2 evaluation protocol uses a list-wise comparison approach where the model selects the best response from multiple candidates, providing insights into the model's preference alignment and ranking capabilities.

## Features

- **Dual Evaluation Modes**: 
  - **Four-way Comparison** (Non-Ties): Selects best response from 4 candidates
  - **Absolute Rating** (Ties): Independent 1-10 scale rating for multiple valid answers
- **Automatic Subset Detection**: Automatically separates and processes Ties vs non-Ties samples
- **Position Bias Mitigation**: Automatically shuffles responses to prevent position-based biases
- **Parallel Processing**: Multi-threaded evaluation with configurable worker count for faster processing
- **Comprehensive Metrics**: Provides accuracy metrics overall and by subset categories
- **Real-time Progress Tracking**: Progress bars for both subset types during evaluation

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

The easiest way to run RewardBench2 evaluation is directly from the command line. The evaluator now supports both standard four-way comparison (non-Ties subsets) and absolute rating evaluation (Ties subsets) with parallel processing:

```bash
# Simplest command with default parameters (evaluates 2 samples by default)
python rm_gallery/gallery/evaluation/rewardbench2.py

# Quick test with 10 samples and 4 parallel workers
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --max_samples 10 \
    --max_workers 4 \
    --result_path "test_results/rewardbench2_test.json"

# Full evaluation with custom parameters
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/rewardbench2_results.json" \
    --max_samples 100 \
    --model "deepseek-chat" \
    --max_workers 8
```

**Expected Output:**
```
Loading data from: data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet
Initializing model: deepseek-chat
Loaded 10 samples for evaluation
Processing 9 non-Ties samples and 1 Ties samples
Using 4 parallel workers
Evaluating non-Ties samples...
[##################################################] 9/9
Evaluating Ties samples...
[##################################################] 1/1
Results saved to: test_results/rewardbench2_test.json
Evaluation completed successfully!
```

### Command Line Parameters

All parameters are optional and have default values:

```bash
# Minimal command with defaults (evaluates 2 samples)
python rm_gallery/gallery/evaluation/rewardbench2.py

# Quick test with parallel processing
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --max_samples 10 \
    --max_workers 4

# Full command with all parameters
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/rewardbench2.json" \
    --max_samples 100 \
    --model "deepseek-chat" \
    --max_workers 8
```

### Evaluation Modes

The evaluator automatically detects and handles two types of subsets:

1. **Non-Ties Subsets** (Standard): Four-way comparison where the model selects the best response from four candidates
2. **Ties Subsets**: Absolute rating where each response is independently rated on a 1-10 scale

The system automatically separates samples by subset type and processes them with the appropriate evaluation method.

### Programmatic Usage

You can also use the evaluation in your Python code:

```python
from rm_gallery.gallery.evaluation.rewardbench2 import main

# Run evaluation with parallel processing
main(
    data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path="data/results/rewardbench2_results.json",
    max_samples=100,
    model="deepseek-chat",
    max_workers=8  # Parallel workers for faster evaluation
)
```

### Environment Variables Setup

For the evaluation to work properly, ensure you have the required environment variables set:

```bash
# Activate your conda environment
conda activate rm

# Set API credentials
export OPENAI_API_KEY="your_api_key_here"
export BASE_URL="your_base_url_here"  # Optional, for custom endpoints

# Run evaluation
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --max_samples 10 \
    --max_workers 4 \
    --result_path "test_results/rewardbench2.json"
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
    )
)

# 5. Run evaluation with parallel processing
results = evaluator.run(
    samples=dataset.get_data_samples(),
    max_workers=8  # Adjust based on API rate limits
)

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
| `data_path` | str | "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" | Path to the RewardBench2 dataset file |
| `result_path` | str | "data/results/rewardbench2.json" | Path to save evaluation results |
| `max_samples` | int | 2 | Maximum number of samples to evaluate (-1 for all) |
| `model` | str/dict | "deepseek-chat" | Model name or configuration dictionary |
| `max_workers` | int | 8 | Number of parallel workers for evaluation (improves performance) |

### Performance Notes

- **Parallel Processing**: The `max_workers` parameter significantly improves evaluation speed. Recommended values:
  - Small tests (≤50 samples): 2-4 workers
  - Medium tests (50-500 samples): 4-8 workers  
  - Large tests (500+ samples): 8-16 workers
- **Automatic Subset Detection**: The evaluator automatically separates Ties and non-Ties samples for appropriate processing
- **Progress Tracking**: Real-time progress bars show completion status for both subset types

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

The evaluation results contain comprehensive metrics for both subset types:

```json
{
    "model": "deepseek-chat",
    "overall_accuracy": {
        "accuracy": 0.75,
        "correct_count": 75,
        "valid_samples": 100,
        "total_samples": 100,
        "ties_samples": 10,
        "non_ties_samples": 90
    },
    "subset_accuracy": {
        "Focus": {
            "accuracy": 0.80,
            "correct_count": 40,
            "valid_samples": 50,
            "total_samples": 50,
            "ties_samples": 0,
            "non_ties_samples": 50
        },
        "Ties": {
            "accuracy": 0.70,
            "correct_count": 7,
            "valid_samples": 10,
            "total_samples": 10,
            "ties_samples": 10,
            "non_ties_samples": 0
        },
        "Math": {
            "accuracy": 0.68,
            "correct_count": 27,
            "valid_samples": 40,
            "total_samples": 40,
            "ties_samples": 0,
            "non_ties_samples": 40
        }
    },
    "non_ties_count": 90,
    "ties_count": 10,
    "total_count": 100,
    "max_workers": 8
}
```

### Key Metrics

- **accuracy**: Proportion of correct predictions (0.0 to 1.0)
- **correct_count**: Number of correctly identified best responses  
- **valid_samples**: Number of successfully processed samples
- **total_samples**: Total number of samples in the subset
- **ties_samples**: Number of Ties subset samples (evaluated with absolute rating)
- **non_ties_samples**: Number of standard four-way comparison samples
- **max_workers**: Number of parallel workers used for evaluation

### Interpreting Results

1. **Overall Accuracy**: Higher values indicate better alignment with human preferences
2. **Subset Performance**: Compare performance across different task categories
   - **Ties subsets**: Use absolute rating (1-10 scale) to handle multiple valid answers
   - **Non-Ties subsets**: Use four-way comparison to select the single best response
3. **Sample Distribution**: Check the balance between Ties and non-Ties samples
4. **Valid Samples**: Ensure most samples were processed successfully (should be close to total_samples)
5. **Performance Analysis**:
   - Ties subsets often have different accuracy patterns due to multiple valid answers
   - Non-Ties subsets provide more direct preference ranking insights

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**:
   ```bash
   # Make sure to set both API key and activate correct environment
   conda activate rm
   export OPENAI_API_KEY="your_api_key_here"
   export BASE_URL="your_base_url_here"  # For custom endpoints
   ```

2. **Import Errors**:
   - Ensure you're using the correct conda environment (`conda activate rm`)
   - Check that OpenAI package version is compatible
   - Verify RM-Gallery installation

3. **Data Path Issues**:
   - Ensure the parquet file exists at the specified path
   - Check file permissions
   - Default path: `data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet`

4. **Connection Issues**:
   - API timeout errors: Reduce `max_workers` to avoid rate limiting
   - Network connectivity issues: Check `BASE_URL` and internet connection
   - API key issues: Verify the key is valid and has sufficient quota

5. **Performance Issues**:
   - Slow evaluation: Increase `max_workers` (but not too high to avoid rate limits)
   - Memory issues: Reduce `max_samples` or `max_workers`
   - Evaluation failures: Check logs for specific error messages

### Performance Optimization

1. **Parallel Processing**: 
   - **Optimal worker count**: Start with 4-8 workers, adjust based on API rate limits
   - **Rate limit awareness**: Too many workers may trigger API rate limiting
   - **Resource monitoring**: Monitor CPU and memory usage

2. **Sample Management**:
   - **Testing**: Start with small samples (10-50) for validation
   - **Production**: Gradually increase to full dataset
   - **Batch processing**: Process in chunks for very large datasets

3. **Model Selection**:
   - **Fast models**: Use lighter models for preliminary testing
   - **Quality vs Speed**: Balance model capability with evaluation speed
   - **Custom configurations**: Adjust temperature and max_tokens for your needs

## Best Practices

1. **Environment Setup**: Always activate the correct conda environment and set API variables
   ```bash
   conda activate rm
   export OPENAI_API_KEY="your_key"
   export BASE_URL="your_endpoint"  # If using custom endpoint
   ```

2. **Progressive Testing**: 
   - Start with 2-10 samples for initial testing
   - Scale to 50-100 for validation
   - Run full evaluation after confirming setup

3. **Parallel Processing**:
   - Start with 4 workers, adjust based on performance
   - Monitor for API rate limit errors
   - Higher worker count ≠ always faster (due to rate limits)

4. **Automatic Handling**:
   - **Position Bias**: Automatically mitigated through response shuffling
   - **Subset Detection**: Ties vs non-Ties samples automatically separated
   - **Progress Tracking**: Real-time progress bars for both subset types

5. **Result Validation**: 
   - Check `valid_samples` vs `total_samples` ratio
   - Analyze both Ties and non-Ties performance separately
   - Review subset-specific accuracy patterns

6. **Reproducibility**: The system handles randomization automatically for bias prevention

## Examples

### Example 1: Quick Testing

```bash
# Minimal test with environment setup
conda activate rm && \
export OPENAI_API_KEY="your_key_here" && \
export BASE_URL="your_endpoint" && \
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --max_samples 10 \
    --max_workers 4 \
    --result_path "test_results/quick_test.json"
```

**Expected Output:**
```
Processing 9 non-Ties samples and 1 Ties samples
Using 4 parallel workers
Evaluating non-Ties samples...
[##################################################] 9/9
Evaluating Ties samples...
[##################################################] 1/1
Results saved to: test_results/quick_test.json
```

### Example 2: Production Evaluation

```bash
# Full evaluation with environment setup and optimized settings
conda activate rm && \
export OPENAI_API_KEY="your_key_here" && \
export BASE_URL="your_endpoint" && \
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --max_samples 500 \
    --model "deepseek-chat" \
    --max_workers 8 \
    --result_path "results/production_eval.json"
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
# Compare different models with proper environment setup
conda activate rm
export OPENAI_API_KEY="your_key_here"
export BASE_URL="your_endpoint"

models=("deepseek-chat" "qwen3-32b" "gpt-4o-mini")

for model in "${models[@]}"; do
    echo "Evaluating with model: $model"
    python rm_gallery/gallery/evaluation/rewardbench2.py \
        --max_samples 100 \
        --model "$model" \
        --max_workers 6 \
        --result_path "results/rewardbench2_${model//\//_}.json"
done

echo "All model evaluations completed!"
```

### Example 4: Comprehensive Evaluation with Error Handling

```bash
#!/bin/bash
# Comprehensive evaluation script with error handling

set -e  # Exit on any error

# Setup environment
conda activate rm
export OPENAI_API_KEY="your_key_here"
export BASE_URL="your_endpoint"

# Create results directory
mkdir -p results

echo "Starting RewardBench2 evaluation..."

# Small test first
echo "Running quick test..."
if python rm_gallery/gallery/evaluation/rewardbench2.py \
    --max_samples 5 \
    --max_workers 2 \
    --result_path "results/test.json"; then
    echo "✅ Quick test passed"
else
    echo "❌ Quick test failed"
    exit 1
fi

# Full evaluation
echo "Running full evaluation..."
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --max_samples 200 \
    --max_workers 8 \
    --result_path "results/rewardbench2_full.json"

echo "🎉 Evaluation completed successfully!"
```

## Next Steps

After running RewardBench2 evaluation:

1. **Analyze Results**: 
   - Review overall accuracy and subset-specific performance
   - Compare Ties vs non-Ties subset accuracy patterns
   - Check `valid_samples` vs `total_samples` ratio

2. **Performance Analysis**:
   - Monitor parallel processing efficiency (adjust `max_workers`)
   - Identify bottlenecks (API rate limits, network issues)
   - Optimize evaluation speed while maintaining quality

3. **Model Comparison**: 
   - Run evaluations with different models using consistent parameters
   - Compare performance across both Ties and non-Ties subsets
   - Analyze cost vs accuracy trade-offs

4. **Production Integration**:
   - Set up automated evaluation pipelines
   - Integrate with your model development workflow
   - Monitor evaluation performance over time

5. **Advanced Usage**:
   - Customize evaluation templates for specific use cases
   - Integrate with other RM-Gallery evaluation tools
   - Scale to larger datasets with optimized parallel processing

For more advanced evaluation scenarios and integration with other benchmarks, check out other evaluation tutorials in the RM-Gallery documentation.


