# Evaluating Reward Models: Overview

## 1. Introduction

Reward model evaluation is crucial for understanding how well your models can judge, rank, and score responses in various scenarios. RM-Gallery provides a comprehensive suite of evaluation benchmarks, each designed to test different aspects of reward model capabilities.

This overview will help you understand:

- **What each benchmark measures** - The specific capabilities and scenarios tested
- **When to use each benchmark** - Guidelines for selecting appropriate evaluation tools
- **How benchmarks complement each other** - Building a complete evaluation strategy
- **Key metrics and interpretation** - Understanding evaluation results

## 2. Available Benchmarks

### 2.1 RewardBench 2.0
**Focus**: Comprehensive multi-domain evaluation

RewardBench 2.0 is the most comprehensive benchmark, covering a wide range of scenarios including:
- Chat interactions
- Safety and harmlessness
- Reasoning capabilities
- Code generation
- Mathematical problem-solving

**Best for**: General-purpose reward model evaluation and comparing models across diverse tasks.

**Key Metrics**:
- Overall accuracy
- Per-category performance
- Domain-specific breakdowns

[→ Learn more about RewardBench](rewardbench2.md)

---

### 2.2 JudgeBench
**Focus**: LLM judge evaluation with multiple protocols

JudgeBench evaluates the ability of reward models to act as judges in pairwise comparisons. It supports multiple judging protocols:
- Vanilla judge
- Arena-Hard style
- AutoJ format
- Prometheus2 evaluation
- Skywork-Critic approach

**Best for**: Testing models specifically designed for comparative evaluation and judge applications.

**Key Metrics**:
- Pairwise accuracy
- Source-wise performance
- Position bias analysis

[→ Learn more about JudgeBench](judgebench.md)

---

### 2.3 RM-Bench
**Focus**: Style-aware evaluation

RM-Bench introduces a unique 3x3 matrix evaluation approach that tests how models handle different response formats:
- Concise responses
- Detailed plain text
- Detailed markdown formatting

**Best for**: Evaluating models that need to handle diverse response styles and formats.

**Key Metrics**:
- Hard/Normal/Easy accuracy tiers
- Style preference patterns
- Multi-domain coverage (chat, code, math, safety)

[→ Learn more about RM-Bench](rmbench.md)

---

### 2.4 RMB (Reward Model Benchmark)
**Focus**: Real-world scenario coverage

RMB provides extensive coverage with 49+ real-world scenarios, evaluating models across:
- **Helpfulness**: brainstorming, classification, code generation, math, reasoning
- **Harmlessness**: safety, toxicity detection, harmful content avoidance

**Best for**: Comprehensive real-world performance assessment across diverse use cases.

**Key Metrics**:
- Pairwise comparison accuracy
- Category-wise breakdowns
- Helpfulness vs. harmlessness balance

[→ Learn more about RMB](rmb.md)

---

### 2.5 Conflict Detector
**Focus**: Logical consistency analysis

The Conflict Detector identifies logical inconsistencies in model judgments:
- **Symmetry conflicts**: Contradictory preferences (A > B and B > A)
- **Transitivity conflicts**: Circular logic (A > B > C but C > A)
- **Cycle detection**: Complex preference loops

**Best for**: Testing model reliability and consistency in decision-making.

**Key Metrics**:
- Conflict rates by type
- Consistency scores
- Logical coherence analysis

[→ Learn more about Conflict Detector](conflict_detector.md)

---

## 3. Choosing the Right Benchmark

### Quick Decision Guide

| Your Goal | Recommended Benchmark(s) |
|-----------|-------------------------|
| General model assessment | RewardBench 2.0 + RMB |
| Judge/evaluator models | JudgeBench |
| Style-sensitive applications | RM-Bench |
| Reliability testing | Conflict Detector |
| Comprehensive validation | All benchmarks |

### Evaluation Strategy

For a thorough evaluation, we recommend a **multi-stage approach**:

1. **Baseline Assessment** (RewardBench 2.0)
   - Establish overall capabilities
   - Identify strength/weakness domains

2. **Specialized Testing** (Based on use case)
   - JudgeBench for judge applications
   - RM-Bench for style-aware tasks
   - RMB for specific scenario coverage

3. **Consistency Validation** (Conflict Detector)
   - Verify logical coherence
   - Test reliability at scale

## 4. Common Evaluation Workflow

Regardless of which benchmark you choose, the typical workflow follows these steps:

### Step 1: Setup Environment
```python
# Install dependencies
pip install rm-gallery

# Configure API keys if using API-based models
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

### Step 2: Download Benchmark Data
```bash
# Each benchmark has its own dataset
mkdir -p data/benchmarks
cd data/benchmarks

# Download specific benchmark (example for RewardBench)
git clone https://github.com/benchmark-repo
```

### Step 3: Configure Evaluation
```python
from rm_gallery.gallery.evaluation import BenchmarkEvaluator

evaluator = BenchmarkEvaluator(
    model_name="your-model",
    benchmark="rewardbench2",
    config={
        "batch_size": 8,
        "max_workers": 4
    }
)
```

### Step 4: Run Evaluation
```python
results = evaluator.evaluate()
```

### Step 5: Analyze Results
```python
# View overall metrics
print(f"Overall Accuracy: {results['accuracy']}")

# Per-category breakdown
for category, score in results['categories'].items():
    print(f"{category}: {score}")

# Export detailed results
results.export("results/evaluation_report.json")
```

## 5. Understanding Metrics

### Accuracy-Based Metrics
Most benchmarks report **accuracy** as the primary metric:
- Percentage of correct preferences/rankings
- Typically broken down by category/domain
- Higher is better (range: 0-100%)

### Ranking Metrics
Some benchmarks use ranking-based evaluation:
- **Spearman correlation**: Rank order correlation
- **Kendall's tau**: Pairwise agreement measure
- Both range from -1 to 1 (higher is better)

### Consistency Metrics
Conflict Detector provides unique consistency metrics:
- **Conflict rate**: Percentage of logical inconsistencies
- **Coherence score**: Overall logical consistency
- Lower conflict rates indicate better reliability

## 6. Best Practices

### Performance Optimization
- Use **parallel processing** for faster evaluation
- Set appropriate **batch sizes** based on your hardware
- Enable **caching** to avoid redundant API calls

### Result Interpretation
- Don't rely on a single metric - examine category breakdowns
- Compare against **baseline models** for context
- Look for consistent patterns across multiple benchmarks

### Iterative Improvement
1. **Identify weaknesses** from evaluation results
2. **Refine your model** (training data, rubrics, architecture)
3. **Re-evaluate** on the same benchmarks
4. **Track progress** over time

## 7. Benchmark Comparison Matrix

| Feature | RewardBench 2.0 | JudgeBench | RM-Bench | RMB | Conflict Detector |
|---------|-----------------|------------|----------|-----|-------------------|
| **Coverage** | Multi-domain | Judge-focused | Style-aware | 49+ scenarios | Consistency |
| **Evaluation Type** | Pairwise | Pairwise | Matrix (3x3) | Pairwise | Logic analysis |
| **Primary Metric** | Accuracy | Accuracy | Accuracy tiers | Accuracy | Conflict rate |
| **Specialized** | General | Judge protocols | Response styles | Real-world | Coherence |
| **Dataset Size** | Large | Medium | Medium | Large | Flexible |
| **Use Case** | General RM | Judge/Evaluator | Format-sensitive | Scenario-specific | Reliability |

## 8. Next Steps

Ready to start evaluating? Choose your benchmark:

- **[RewardBench 2.0](rewardbench2.md)** - Start here for comprehensive evaluation
- **[JudgeBench](judgebench.md)** - For judge/evaluator applications
- **[RM-Bench](rmbench.md)** - For style-aware testing
- **[RMB](rmb.md)** - For extensive scenario coverage
- **[Conflict Detector](conflict_detector.md)** - For consistency testing

Each benchmark page provides detailed setup instructions, code examples, and result interpretation guidelines.

---

## Additional Resources

- **[Building RM Overview](../building_rm/overview.ipynb)** - Learn how to build reward models
- **[RM Library](../../library/rm_library.md)** - Pre-built reward models
- **[Best Practices](../building_rm/benchmark_practices.ipynb)** - Evaluation best practices

