# RM-Bench Evaluation Tutorial

## Overview

RM-Bench is a specialized benchmark for evaluating reward models of language models with a focus on **subtlety and style**. Unlike traditional benchmarks that only consider binary comparisons, RM-Bench evaluates models across multiple response styles and formats, providing a more nuanced assessment of reward model capabilities.

The benchmark introduces a unique 3x3 matrix evaluation approach that compares responses with different levels of style sophistication:
- **Concise style**: Brief, direct responses
- **Detailed plain text**: Comprehensive responses in plain format
- **Detailed markdown**: Well-formatted responses with markdown styling

## Key Features

- **Style-Aware Evaluation**: Assesses model preferences across different response formats
- **Multi-Domain Coverage**: Includes chat, code, math, safety-refuse, and safety-response domains
- **Sophisticated Metrics**: Three accuracy types (hard, normal, easy) for comprehensive analysis
- **Parallel Processing**: Efficient evaluation with concurrent response comparisons
- **Matrix-Based Analysis**: 3x3 comparison matrix for detailed preference patterns

## Quick Start

### Step 1: Download Dataset

```bash
# Download the RM-Bench dataset
# Note: Ensure you have the dataset file at the specified path
mkdir -p data/benchmarks
cd data/benchmarks
git clone https://github.com/THU-KEG/RM-Bench.git
cd ../../
# Download total_dataset.json to data/benchmarks/RM-Bench/
# You can get the dataset from the official repository or other sources
```

### Step 2: Verify Installation

```bash
# Check if the module can be imported
python -c "from rm_gallery.gallery.evaluation.rmbench import main; print('RM-Bench evaluation module loaded successfully')"
```

### Step 3: Basic Usage

```bash
# Run evaluation on a sample dataset
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_results.json" \
    --max_samples=10 \
    --model="gpt-4o-mini"
```

### Step 4: Check Results

```bash
# View evaluation results
cat data/results/rmbench_results.json
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

Evaluate a small subset of the RM-Bench dataset:

```bash
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_basic.json" \
    --max_samples=50 \
    --model="gpt-4o-mini" \
    --max_workers=4
```

### Full Dataset Evaluation

Run evaluation on the complete dataset:

```bash
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_full.json" \
    --max_samples=1000 \
    --model="gpt-4o" \
    --max_workers=8
```

### High-Performance Evaluation

For large-scale evaluation with maximum parallelism:

```bash
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_performance.json" \
    --max_samples=2000 \
    --model="claude-3-5-sonnet-20241022" \
    --max_workers=16
```

### Model Comparison

Compare different models on the same dataset:

```bash
# Evaluate GPT-4o
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_gpt4o.json" \
    --model="gpt-4o" \
    --max_samples=200

# Evaluate Claude
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_claude.json" \
    --model="claude-3-5-sonnet-20241022" \
    --max_samples=200

# Evaluate Qwen
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_qwen.json" \
    --model="qwen3-32b" \
    --max_samples=200
```

## Configuration Parameters

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | `"data/benchmarks/RM-Bench/total_dataset.json"` | Path to RM-Bench dataset file |
| `result_path` | str | `"data/results/rmbench.json"` | Output file path for results |
| `max_samples` | int | `10` | Maximum number of samples to evaluate |
| `model` | str/dict | `"qwen3-32b"` | Model identifier or configuration dictionary |
| `max_workers` | int | `8` | Number of parallel processing workers |

### Advanced Configuration

For advanced usage with custom model parameters:

```bash
# Custom model configuration
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_custom.json" \
    --model='{"model": "gpt-4o", "temperature": 0.1, "max_tokens": 2048, "timeout": 120}' \
    --max_samples=100 \
    --max_workers=6
```

## Dataset Structure

The RM-Bench dataset follows a specific format:

```json
{
    "id": "unique_sample_identifier",
    "prompt": "Input prompt given to the model",
    "chosen": [
        "resp_1", // Chosen response with concise style
        "resp_2", // Chosen response with detailed plain text style
        "resp_3"  // Chosen response with detailed markdown style
    ],
    "rejected": [
        "resp_1", // Rejected response with concise style
        "resp_2", // Rejected response with detailed plain text style
        "resp_3"  // Rejected response with detailed markdown style
    ],
    "domain": "chat" // Domain: chat, code, math, safety-refuse, safety-response
}
```

### Style Categories

1. **Concise (Index 0)**: Brief, direct responses
2. **Detailed Plain (Index 1)**: Comprehensive responses in plain text format
3. **Detailed Markdown (Index 2)**: Well-formatted responses with markdown styling

## Evaluation Methodology

### 3x3 Matrix Comparison

RM-Bench uses a unique 3x3 matrix evaluation approach:

```
              Rejected Responses
              [0]  [1]  [2]
Chosen    [0]  *    *    *
Responses [1]  *    *    *
          [2]  *    *    *
```

Each cell `(i,j)` represents the comparison between chosen response `i` and rejected response `j`.

### Accuracy Metrics

RM-Bench provides three distinct accuracy measurements:

1. **Hard Accuracy**: Upper-right triangle of the matrix
   - Compares chosen responses with less sophisticated style vs rejected responses with more sophisticated style
   - Most challenging scenario for reward models

2. **Normal Accuracy**: Diagonal of the matrix
   - Compares responses with the same style level
   - Standard preference evaluation

3. **Easy Accuracy**: Lower-left triangle of the matrix
   - Compares chosen responses with more sophisticated style vs rejected responses with less sophisticated style
   - Easiest scenario for reward models

## Output Format

### Results Structure

The evaluation generates a JSON file with the following structure:

```json
{
  "model": "gpt-4o-mini",
  "overall_accuracy": {
    "hard_acc": 0.65,
    "normal_acc": 0.78,
    "easy_acc": 0.85,
    "overall_acc": 0.76,
    "valid_samples": 95,
    "total_samples": 100,
    "acc_matrix": [
      [0.75, 0.68, 0.62],
      [0.82, 0.78, 0.71],
      [0.88, 0.85, 0.81]
    ]
  },
  "subset_accuracy": {
    "chat": {
      "hard_acc": 0.70,
      "normal_acc": 0.80,
      "easy_acc": 0.88
    },
    "code": {
      "hard_acc": 0.62,
      "normal_acc": 0.75,
      "easy_acc": 0.83
    }
  }
}
```

### Metrics Explanation

- **hard_acc**: Accuracy when chosen has simpler style than rejected
- **normal_acc**: Accuracy when chosen and rejected have same style
- **easy_acc**: Accuracy when chosen has more sophisticated style than rejected
- **overall_acc**: Average of all matrix cells
- **valid_samples**: Number of successfully processed samples
- **total_samples**: Total number of input samples
- **acc_matrix**: 3x3 matrix showing detailed comparison results

## Expected Output

When running the evaluation, you should see output similar to:

```bash
$ python rm_gallery/gallery/evaluation/rmbench.py --max_samples=10

INFO - Starting RM-Bench evaluation...
INFO - Loading dataset from: data/benchmarks/RM-Bench/total_dataset.json
INFO - Model: gpt-4o-mini
INFO - Processing 10 samples with 3x3 matrix comparisons...
INFO - Evaluation completed!
INFO - Results saved to: data/results/rmbench.json
INFO - Hard accuracy: 65.0%
INFO - Normal accuracy: 78.0%
INFO - Easy accuracy: 85.0%
INFO - Overall accuracy: 76.0%
INFO - Valid samples: 10/10
```

## Domain-Specific Analysis

### Domain Categories

RM-Bench covers five main domains:

1. **Chat**: General conversational responses
2. **Code**: Programming and technical content
3. **Math**: Mathematical problem-solving
4. **Safety-refuse**: Appropriate refusal of harmful requests
5. **Safety-response**: Safe handling of sensitive topics

### Per-Domain Evaluation

You can analyze results by domain:

```python
import json

# Load results
with open("data/results/rmbench_results.json", "r") as f:
    results = json.load(f)

# Analyze domain performance
for domain, metrics in results["subset_accuracy"].items():
    print(f"Domain: {domain}")
    print(f"  Hard accuracy: {metrics['hard_acc']:.2%}")
    print(f"  Normal accuracy: {metrics['normal_acc']:.2%}")
    print(f"  Easy accuracy: {metrics['easy_acc']:.2%}")
    print()
```

## Integration with Other Benchmarks

### Combining with JudgeBench and RMB

Run comprehensive evaluation across multiple benchmarks:

```bash
# Run RM-Bench evaluation
python rm_gallery/gallery/evaluation/rmbench.py \
    --data_path="data/benchmarks/RM-Bench/total_dataset.json" \
    --result_path="data/results/rmbench_results.json" \
    --model="gpt-4o-mini"

# Run JudgeBench evaluation
python rm_gallery/gallery/evaluation/judgebench.py \
    --data_path="data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
    --result_path="data/results/judgebench_results.json" \
    --model="gpt-4o-mini"

# Run RMB evaluation
python rm_gallery/gallery/evaluation/rmb.py \
    --data_path="data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set" \
    --result_path="data/results/rmb_results.json" \
    --model="gpt-4o-mini"
```

### Comparative Analysis

```python
# Compare results across benchmarks
import json

benchmarks = ["rmbench", "judgebench", "rmb"]
results = {}

for benchmark in benchmarks:
    with open(f"data/results/{benchmark}_results.json", "r") as f:
        results[benchmark] = json.load(f)

# Print comparison
for benchmark, result in results.items():
    if benchmark == "rmbench":
        accuracy = result["overall_accuracy"]["overall_acc"]
    elif benchmark == "judgebench":
        accuracy = result["overall_accuracy"]["accuracy"]
    else:  # rmb
        accuracy = result["overall_accuracy"]["accuracy"]
    
    print(f"{benchmark}: {accuracy:.2%}")
```

## Troubleshooting

### Common Issues

1. **Dataset format errors**
   ```bash
   # Verify dataset structure
   python -c "import json; data = json.load(open('data/benchmarks/RM-Bench/total_dataset.json')); print(f'Dataset loaded: {len(data)} samples')"
   ```

2. **Model API errors**
   ```bash
   # Check API key configuration
   echo $OPENAI_API_KEY
   ```

3. **Memory issues with large datasets**
   ```bash
   # Reduce max_samples and max_workers
   python rm_gallery/gallery/evaluation/rmbench.py --max_samples=10 --max_workers=2
   ```

4. **Invalid comparison matrix**
   ```bash
   # Check for samples with invalid matrix results
   python -c "import json; results = json.load(open('data/results/rmbench_results.json')); print(f\"Valid samples: {results['overall_accuracy']['valid_samples']}\")"
   ```

### Performance Optimization

- **Parallel Processing**: Increase `max_workers` for better throughput
- **Batch Size**: Adjust `max_samples` based on available memory
- **Model Selection**: Use efficient models for large-scale evaluation

### Error Resolution

If you encounter evaluation errors:

1. Check the 3x3 matrix completion rate
2. Verify dataset sample format
3. Confirm model response parsing
4. Reduce concurrency if rate-limited

## Advanced Usage

### Custom Template Modification

You can customize the evaluation template by modifying the `RMBenchTemplate.format()` method:

```python
# Example: Add domain-specific evaluation criteria
def format(cls, query: str, answers: List[str], domain: str = None, **kwargs) -> str:
    # Add domain-specific instructions
    domain_instructions = {
        "code": "Focus on code correctness, efficiency, and readability.",
        "math": "Evaluate mathematical accuracy and solution clarity.",
        "safety": "Assess safety and appropriateness of responses."
    }
    
    instruction = domain_instructions.get(domain, "")
    # ... rest of template logic
```

### Matrix Analysis

Analyze the 3x3 comparison matrix in detail:

```python
import numpy as np
import json

# Load results
with open("data/results/rmbench_results.json", "r") as f:
    results = json.load(f)

matrix = np.array(results["overall_accuracy"]["acc_matrix"])

# Analyze preference patterns
print("Comparison Matrix:")
print(matrix)
print(f"Hard accuracy (upper right): {np.sum(np.triu(matrix, 1)) / 3:.2%}")
print(f"Normal accuracy (diagonal): {np.mean(np.diag(matrix)):.2%}")
print(f"Easy accuracy (lower left): {np.sum(np.tril(matrix, -1)) / 3:.2%}")
```

## Best Practices

1. **Start with Small Samples**: Begin with 10-20 samples for testing
2. **Monitor Matrix Completion**: Check valid_samples vs total_samples ratio
3. **Analyze by Domain**: Review performance across different content types
4. **Compare Accuracy Types**: Focus on hard_acc as the most challenging metric
5. **Use Appropriate Models**: Ensure models can handle style differentiation

This tutorial provides a comprehensive guide to using the RM-Bench evaluation system for assessing reward models with style and subtlety considerations. The unique 3x3 matrix approach offers deeper insights into model preferences across different response formats and sophistication levels. 