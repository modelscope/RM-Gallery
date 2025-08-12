# RMB Benchmark Evaluation Tutorial

## Overview

RMB (Reward Model Benchmark) is a comprehensive evaluation framework for reward models in LLM alignment. It covers over 49 real-world scenarios and includes both pairwise and Best-of-N (BoN) evaluations to better reflect the effectiveness of reward models in guiding alignment optimization.

The benchmark evaluates models across two main dimensions:
- **Helpfulness**: Including brainstorming, chat, classification, code generation, math, reasoning, and more
- **Harmlessness**: Focusing on safety, toxicity detection, and harmful content avoidance

## Features

- **Comprehensive Coverage**: 49+ real-world scenarios across helpfulness and harmlessness
- **Pairwise Comparisons**: Direct comparison between model responses
- **Category Analysis**: Detailed breakdown by scenario categories
- **Parallel Processing**: Efficient evaluation with configurable worker threads
- **Flexible Model Support**: Compatible with various LLM APIs

## Quick Start

### Step 1: Download Dataset

```bash
# Create benchmarks directory and clone dataset
mkdir -p data/benchmarks
cd data/benchmarks
git clone https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark.git
cd ../../
```

### Step 2: Verify Installation

Ensure you have the required environment set up:

```bash
# Check if the module can be imported
python -c "from rm_gallery.gallery.evaluation.rmb import main; print('RMB evaluation module loaded successfully')"
```

### Step 3: Basic Usage

```bash
# Run evaluation on a sample dataset
python rm_gallery/gallery/evaluation/rmb.py \
    --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set" \
    --result_path="data/results/rmb_results.json" \
    --max_samples=10 \
    --model="gpt-4o-mini"
```

### Step 4: Check Results

```bash
# View evaluation results
cat data/results/rmb_results.json
```

## Installation and Environment Setup

### Prerequisites

```bash
# Install required dependencies
pip install fire loguru numpy pydantic
```

### Environment Variables

Set up your API keys:

```bash
# For OpenAI models
export OPENAI_API_KEY="your-api-key"

# For other providers
export ANTHROPIC_API_KEY="your-anthropic-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

### Verify Setup

```bash
# Test model connection
python -c "from rm_gallery.core.model.openai_llm import OpenaiLLM; llm = OpenaiLLM(model='gpt-4o-mini'); print('Model connection successful')"
```

## Usage Examples

### Basic Evaluation

Evaluate a small subset of the RMB dataset:

```bash
python rm_gallery/gallery/evaluation/rmb.py \
    --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set" \
    --result_path="data/results/rmb_basic.json" \
    --max_samples=50 \
    --model="gpt-4o-mini" \
    --max_workers=4
```

### Full Dataset Evaluation

Run evaluation on the complete dataset:

```bash
python rm_gallery/gallery/evaluation/rmb.py \
    --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set" \
    --result_path="data/results/rmb_full.json" \
    --max_samples=1000 \
    --model="gpt-4o" \
    --max_workers=8
```

### Specific Category Evaluation

Evaluate only helpfulness scenarios:

```bash
python rm_gallery/gallery/evaluation/rmb.py \
    --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set/Helpfulness" \
    --result_path="data/results/rmb_helpfulness.json" \
    --max_samples=200 \
    --model="claude-3-5-sonnet-20241022"
```

### Model Comparison

Compare different models on the same dataset:

```bash
# Evaluate GPT-4o
python rm_gallery/gallery/evaluation/rmb.py \
    --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set" \
    --result_path="data/results/rmb_gpt4o.json" \
    --model="gpt-4o" \
    --max_samples=100

# Evaluate Claude
python rm_gallery/gallery/evaluation/rmb.py \
    --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set" \
    --result_path="data/results/rmb_claude.json" \
    --model="claude-3-5-sonnet-20241022" \
    --max_samples=100
```

## Configuration Parameters

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | `"data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set"` | Path to RMB dataset |
| `result_path` | str | `"data/results/rmb.json"` | Output file path |
| `max_samples` | int | `10` | Maximum number of samples to evaluate |
| `model` | str/dict | `"qwen3-32b"` | Model identifier or configuration |
| `max_workers` | int | `8` | Number of parallel processing workers |

### Advanced Configuration

For advanced usage, you can pass model configuration as a dictionary:

```python
# Example with custom model configuration
model_config = {
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 2048,
    "timeout": 60
}

# Run with custom configuration
python rm_gallery/gallery/evaluation/rmb.py \
    --model='{"model": "gpt-4o", "temperature": 0.1, "max_tokens": 2048}'
```

## Dataset Structure

The RMB dataset is organized as follows:

```
RMB_dataset/
├── Pairwise_set/
│   ├── Helpfulness/
│   │   ├── Brainstorming/
│   │   ├── Chat/
│   │   ├── Classification/
│   │   ├── Code/
│   │   ├── Math/
│   │   └── ...
│   └── Harmlessness/
│       ├── Safety/
│       ├── Toxicity/
│       └── ...
└── Best_of_N_set/
    └── ...
```

## Output Format

### Results Structure

The evaluation generates a JSON file with the following structure:

```json
{
  "model": "gpt-4o-mini",
  "overall_accuracy": {
    "accuracy": 0.75,
    "correct_count": 75,
    "valid_samples": 100,
    "total_samples": 100,
    "choice_distribution": {
      "0": 45,
      "1": 55
    }
  },
  "subset_accuracy": {
    "Helpfulness/Brainstorming": {
      "accuracy": 0.80,
      "correct_count": 20,
      "valid_samples": 25,
      "total_samples": 25
    },
    "Harmlessness/Safety": {
      "accuracy": 0.70,
      "correct_count": 14,
      "valid_samples": 20,
      "total_samples": 20
    }
  }
}
```

### Metrics Explanation

- **accuracy**: Percentage of correct preference selections
- **correct_count**: Number of samples where the model chose the preferred response
- **valid_samples**: Number of successfully processed samples
- **total_samples**: Total number of input samples
- **choice_distribution**: Distribution of selected responses (0 for first, 1 for second)

## Expected Output

When running the evaluation, you should see output similar to:

```bash
$ python rm_gallery/gallery/evaluation/rmb.py --max_samples=10

INFO - Starting RMB evaluation...
INFO - Loading dataset from: data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set
INFO - Model: gpt-4o-mini
INFO - Processing 10 samples...
INFO - Evaluation completed!
INFO - Results saved to: data/results/rmb.json
INFO - Overall accuracy: 75.0%
INFO - Valid samples: 10/10
```

## Integration with Other Tools

### Combining with JudgeBench

You can run both RMB and JudgeBench evaluations for comprehensive assessment:

```bash
# Run RMB evaluation
python rm_gallery/gallery/evaluation/rmb.py \
    --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set" \
    --result_path="data/results/rmb_results.json" \
    --model="gpt-4o-mini"

# Run JudgeBench evaluation
python rm_gallery/gallery/evaluation/judgebench.py \
    --data_path="data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
    --result_path="data/results/judgebench_results.json" \
    --model="gpt-4o-mini"
```

### Batch Processing

For processing multiple categories in batch:

```bash
#!/bin/bash
# batch_rmb_eval.sh

categories=(
    "Helpfulness/Brainstorming"
    "Helpfulness/Chat"
    "Helpfulness/Code"
    "Harmlessness/Safety"
)

for category in "${categories[@]}"; do
    echo "Processing category: $category"
    python rm_gallery/gallery/evaluation/rmb.py \
        --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set/$category" \
        --result_path="data/results/rmb_${category//\//_}.json" \
        --model="gpt-4o-mini" \
        --max_samples=50
done
```

## Troubleshooting

### Common Issues

1. **Dataset not found**
   ```bash
# Ensure you've cloned the repository
ls data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/
```

2. **Model API errors**
   ```bash
   # Check API key configuration
   echo $OPENAI_API_KEY
   ```

3. **Memory issues with large datasets**
   ```bash
   # Reduce max_samples and max_workers
   python rm_gallery/gallery/evaluation/rmb.py --max_samples=10 --max_workers=2
   ```

4. **Import errors**
   ```bash
   # Verify installation
   python -c "import rm_gallery.gallery.evaluation.rmb"
   ```

### Performance Optimization

- **Parallel Processing**: Increase `max_workers` based on your system capacity
- **Batch Size**: Adjust `max_samples` to balance memory usage and processing time
- **Model Selection**: Use faster models for initial testing, more capable models for final evaluation

### Error Resolution

If you encounter evaluation errors:

1. Check the error logs in the output
2. Verify dataset file integrity
3. Confirm model API accessibility
4. Reduce concurrency if rate-limited

## Best Practices

1. **Start Small**: Begin with a small `max_samples` value for testing
2. **Monitor Resources**: Watch CPU and memory usage during evaluation
3. **Save Intermediate Results**: Use different output paths for different experiments
4. **Validate Results**: Review accuracy metrics and choice distributions
5. **Document Experiments**: Keep track of model configurations and results

## Advanced Features

### Custom Templates

The RMB evaluation uses structured templates for consistent evaluation. You can modify the template in `RMBTemplate.format()` method for custom evaluation criteria.

### Result Analysis

After evaluation, you can analyze results programmatically:

```python
import json

# Load results
with open("data/results/rmb_results.json", "r") as f:
    results = json.load(f)

# Analyze category performance
for category, metrics in results["subset_accuracy"].items():
    print(f"{category}: {metrics['accuracy']:.2%}")
```

This tutorial provides a comprehensive guide to using the RMB benchmark evaluation system. For additional support or questions, refer to the original RMB repository documentation. 