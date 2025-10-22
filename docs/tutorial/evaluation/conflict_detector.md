# Conflict Detector Tutorial

## Overview

The Conflict Detector is a sophisticated evaluation tool designed to identify **logical inconsistencies** in AI model comparisons. Unlike traditional evaluation metrics that focus on accuracy, the Conflict Detector analyzes the **coherence and consistency** of model preferences across multiple response comparisons.

This tool is particularly valuable for:
- **Reward Model Evaluation**: Assessing the consistency of reward models in ranking responses
- **Judge Model Analysis**: Detecting contradictions in AI judges' decision-making
- **Preference Learning**: Understanding stability in preference-based systems
- **Model Reliability**: Quantifying the logical coherence of model outputs

## Key Features

- **Multi-Type Conflict Detection**: Identifies symmetry, transitivity, and cycle conflicts
- **Comprehensive Analysis**: Provides detailed statistics and conflict visualization
- **Pairwise Comparison**: Evaluates all possible response pairs systematically
- **Parallel Processing**: Efficient evaluation with configurable worker threads
- **Statistical Reporting**: Generates detailed conflict rates and consistency metrics

## Conflict Types Explained

### 1. Symmetry Conflicts
**Definition**: When a model simultaneously prefers A over B and B over A
```
Example: Model says "Response A > Response B" AND "Response B > Response A"
```
**Significance**: Indicates fundamental inconsistency in judgment criteria

### 2. Transitivity Conflicts
**Definition**: When preference chains are broken (A>B>C but A≤C)
```
Example: A > B, B > C, but A ≤ C
```
**Significance**: Shows logical reasoning failures in comparative evaluation

### 3. Cycle Conflicts
**Definition**: When circular preferences exist (A>B>C>A)
```
Example: Response A > Response B > Response C > Response A
```
**Significance**: Represents the most severe form of logical inconsistency

## Quick Start

### Step 1: Download RewardBench2 Dataset

```bash
# Create benchmarks directory and download dataset
mkdir -p data/benchmarks
cd data/benchmarks

# Download RewardBench2 dataset
git clone https://huggingface.co/datasets/allenai/reward-bench-2

cd ../../
```

### Step 2: Verify Installation

```bash
# Check if the module can be imported
python -c "from rm_gallery.gallery.evaluation.conflict_detector import main; print('Conflict Detector module loaded successfully')"
```

### Step 3: Basic Usage

```bash
# Run conflict detection on a sample dataset
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_detection.json" \
    --max_samples=10 \
    --model="gpt-4o-mini"
```

### Step 4: Check Results

```bash
# View conflict detection results
cat data/results/conflict_detection.json
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

# Check dataset accessibility
python -c "import pandas as pd; df = pd.read_parquet('data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet'); print(f'Dataset loaded: {len(df)} samples')"
```

## Usage Examples

### Basic Conflict Detection

Analyze a small subset for quick evaluation:

```bash
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_basic.json" \
    --max_samples=50 \
    --model="gpt-4o-mini" \
    --max_workers=4
```

### Large-Scale Analysis

Run comprehensive conflict detection:

```bash
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_comprehensive.json" \
    --max_samples=500 \
    --model="gpt-4o" \
    --max_workers=8
```

### High-Performance Detection

For maximum throughput with powerful models:

```bash
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_performance.json" \
    --max_samples=1000 \
    --model="claude-3-5-sonnet-20241022" \
    --max_workers=16
```

### Model Comparison Analysis

Compare consistency across different models:

```bash
# Analyze GPT-4o consistency
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_gpt4o.json" \
    --model="gpt-4o" \
    --max_samples=200

# Analyze Claude consistency
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_claude.json" \
    --model="claude-3-5-sonnet-20241022" \
    --max_samples=200

# Analyze Qwen consistency
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_qwen.json" \
    --model="qwen2.5-14b-instruct" \
    --max_samples=200
```

## Configuration Parameters

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | `"data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet"` | Path to RewardBench2 dataset |
| `result_path` | str | `"data/results/conflict.json"` | Output file path for results |
| `max_samples` | int | `10` | Maximum number of samples to evaluate |
| `model` | str/dict | `"qwen2.5-14b-instruct"` | Model identifier or configuration |
| `max_workers` | int | `8` | Number of parallel processing workers |

### Advanced Configuration

For custom model parameters:

```bash
# Custom model configuration with specific parameters
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_custom.json" \
    --model='{"model": "gpt-4o", "temperature": 0.1, "max_tokens": 1024, "timeout": 90}' \
    --max_samples=100 \
    --max_workers=6
```

## Understanding the Results

### Output Format

The evaluation generates a comprehensive JSON report:

```json
{
  "overall_conflict_rate": 0.25,
  "symmetry_conflict_rate": 0.12,
  "transitivity_conflict_rate": 0.08,
  "cycle_conflict_rate": 0.05,
  "conflicts_per_sample": 2.3,
  "consistent_samples_ratio": 0.75,
  "total_samples": 100,
  "valid_samples": 98,
  "total_conflicts": 225,
  "conflict_distribution": {
    "symmetry": 120,
    "transitivity": 78,
    "cycle": 27
  }
}
```

### Metrics Explanation

- **overall_conflict_rate**: Average number of conflicts per sample
- **symmetry_conflict_rate**: Proportion of samples with symmetry conflicts
- **transitivity_conflict_rate**: Proportion of samples with transitivity violations
- **cycle_conflict_rate**: Proportion of samples with circular preferences
- **conflicts_per_sample**: Average total conflicts across all samples
- **consistent_samples_ratio**: Percentage of samples with no conflicts
- **conflict_distribution**: Count of each conflict type

### Interpretation Guidelines

#### Excellent Consistency (conflict_rate < 0.1)
- Model demonstrates high logical coherence
- Suitable for production use in preference learning
- Minimal contradictions in judgment

#### Good Consistency (0.1 ≤ conflict_rate < 0.3)
- Acceptable level of inconsistency
- May require additional training or fine-tuning
- Monitor for specific conflict patterns

#### Poor Consistency (conflict_rate ≥ 0.3)
- Significant logical inconsistencies
- Requires substantial model improvement
- Not recommended for critical applications

## Expected Output

When running the conflict detector, you should see:

```bash
$ python rm_gallery/gallery/evaluation/conflict_detector.py --max_samples=10

INFO - Starting conflict detection analysis...
INFO - Loading RewardBench2 dataset from: data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet
INFO - Model: gpt-4o-mini
INFO - Processing 10 samples with pairwise comparisons...
INFO - Detected 23 total conflicts across samples
INFO - Symmetry conflicts: 12 (52.2%)
INFO - Transitivity conflicts: 8 (34.8%)
INFO - Cycle conflicts: 3 (13.0%)
INFO - Consistent samples: 7/10 (70.0%)
INFO - Results saved to: data/results/conflict_detection.json
```

## Practical Applications

### 1. Reward Model Validation

```bash
# Evaluate reward model consistency
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/reward_model_consistency.json" \
    --model="your-reward-model" \
    --max_samples=500
```

### 2. Judge Model Analysis

```bash
# Analyze AI judge consistency
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/judge_consistency.json" \
    --model="judge-model" \
    --max_samples=300
```

### 3. Model Comparison Study

```python
# Compare consistency across models
import json
import matplotlib.pyplot as plt

models = ["gpt-4o", "claude-3-5-sonnet-20241022", "qwen2.5-14b-instruct"]
conflict_rates = []

for model in models:
    with open(f"data/results/conflict_{model.replace('-', '_')}.json", "r") as f:
        results = json.load(f)
        conflict_rates.append(results["overall_conflict_rate"])

plt.bar(models, conflict_rates)
plt.ylabel("Conflict Rate")
plt.title("Model Consistency Comparison")
plt.show()
```

## Integration with Other Evaluations

### Combining with Other Benchmarks

```bash
# Run comprehensive evaluation pipeline
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/conflict_results.json" \
    --model="gpt-4o-mini"

# Also run standard benchmarks
python rm_gallery/gallery/evaluation/judgebench.py \
    --data_path="data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
    --result_path="data/results/judgebench_results.json" \
    --model="gpt-4o-mini"
```

### Batch Processing Pipeline

```bash
#!/bin/bash
# batch_conflict_analysis.sh

models=(
    "gpt-4o-mini"
    "gpt-4o"
    "claude-3-5-sonnet-20241022"
    "qwen2.5-14b-instruct"
)

for model in "${models[@]}"; do
    echo "Analyzing conflicts for model: $model"
    python rm_gallery/gallery/evaluation/conflict_detector.py \
        --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
        --result_path="data/results/conflict_${model//[-.]/_}.json" \
        --model="$model" \
        --max_samples=100
done
```

## Troubleshooting

### Common Issues

1. **Dataset access errors**
   ```bash
   # Verify dataset download
   ls -la data/benchmarks/reward-bench-2/data/
   ```

2. **Model API errors**
   ```bash
   # Check API key configuration
   echo $OPENAI_API_KEY
   ```

3. **Memory issues with large datasets**
   ```bash
   # Reduce sample size and workers
   python rm_gallery/gallery/evaluation/conflict_detector.py --max_samples=20 --max_workers=2
   ```

4. **Comparison matrix errors**
   ```bash
   # Check for incomplete comparisons
   python -c "
   import json
   with open('data/results/conflict_detection.json') as f:
       results = json.load(f)
       print(f'Valid samples: {results[\"valid_samples\"]}/{results[\"total_samples\"]}')
   "
   ```

### Performance Optimization

- **Parallel Processing**: Increase `max_workers` for better throughput
- **Sample Size**: Start with small samples for testing
- **Model Selection**: Use efficient models for large-scale analysis
- **Batch Processing**: Process multiple models in parallel

### Error Resolution

If you encounter evaluation errors:

1. Check pairwise comparison completion rates
2. Verify dataset sample format and quality
3. Confirm model response parsing accuracy
4. Reduce concurrency if rate-limited

## Advanced Usage

### Custom Conflict Analysis

```python
# Analyze specific conflict patterns
import json
import numpy as np

with open("data/results/conflict_detection.json", "r") as f:
    results = json.load(f)

# Calculate conflict severity distribution
conflict_types = results["conflict_distribution"]
total_conflicts = sum(conflict_types.values())

for conflict_type, count in conflict_types.items():
    percentage = (count / total_conflicts) * 100
    print(f"{conflict_type}: {count} ({percentage:.1f}%)")
```

### Matrix Visualization

```python
# Visualize comparison matrix patterns
import numpy as np
import matplotlib.pyplot as plt

# Load detailed sample data (if available)
# This would require additional data collection during evaluation
def visualize_conflict_matrix(comparison_matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(comparison_matrix, cmap='RdBu', center=0)
    plt.colorbar(label='Comparison Score')
    plt.title('Response Comparison Matrix')
    plt.xlabel('Response Index')
    plt.ylabel('Response Index')
    plt.show()
```

## Research Applications

### 1. Model Consistency Studies

Use conflict detection to study model behavior across different domains:

```bash
# Analyze consistency across different prompt types
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path="data/results/domain_consistency.json" \
    --max_samples=200
```

### 2. Training Data Quality Assessment

Evaluate training data consistency:

```bash
# Check training data for logical inconsistencies
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="training_data.parquet" \
    --result_path="data/results/training_data_conflicts.json" \
    --max_samples=1000
```

### 3. Preference Learning Evaluation

Assess preference learning model quality:

```bash
# Evaluate preference learning models
python rm_gallery/gallery/evaluation/conflict_detector.py \
    --data_path="preference_data.parquet" \
    --result_path="data/results/preference_conflicts.json" \
    --max_samples=500
```

## Best Practices

1. **Start Small**: Begin with 10-20 samples for initial testing
2. **Monitor Metrics**: Focus on consistent_samples_ratio as key indicator
3. **Analyze Patterns**: Look for specific conflict types in your domain
4. **Iterate Models**: Use results to guide model improvement
5. **Cross-Validate**: Test across multiple datasets and domains

## Significance and Impact

The Conflict Detector serves several critical functions in AI evaluation:

### 1. **Quality Assurance**
- Identifies models with systematic logical flaws
- Prevents deployment of inconsistent AI systems
- Ensures reliable preference learning

### 2. **Model Development**
- Guides training data curation
- Informs model architecture decisions
- Supports iterative improvement processes

### 3. **Research Insights**
- Reveals patterns in model reasoning
- Enables comparative analysis across architectures
- Supports theoretical understanding of AI consistency

### 4. **Production Readiness**
- Validates models before deployment
- Establishes consistency benchmarks
- Monitors model degradation over time

This tutorial provides a comprehensive guide to using the Conflict Detector for evaluating AI model consistency and logical coherence. The tool's ability to identify and quantify different types of conflicts makes it invaluable for ensuring reliable AI systems in production environments.