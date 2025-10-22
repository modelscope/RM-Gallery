# Training Reward Models: Overview

## 1. Introduction

Training reward models is a critical step in building AI systems that can evaluate and rank responses according to human preferences. RM-Gallery provides comprehensive tools and frameworks for training reward models from scratch or fine-tuning existing models.

This section will guide you through:

- **Data preparation** - Loading, annotating, and processing training data
- **Training approaches** - Different methods for reward model training
- **Framework integration** - Using VERL for efficient distributed training
- **Best practices** - Optimization strategies and evaluation methods

## 2. Training Workflow

The complete reward model training process follows these stages:

```
┌─────────────────┐
│  1. Data Prep   │  Load and process preference data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Annotation  │  Label and structure training samples
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Training    │  Train model using selected approach
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Evaluation  │  Validate model performance
└─────────────────┘
```

## 3. Data Preparation

### 3.1 Data Pipeline
The data pipeline provides flexible loading and processing capabilities:
- **Multiple source support**: HuggingFace, local files, databases
- **Format conversion**: Automatic schema mapping
- **Batch processing**: Efficient handling of large datasets
- **Quality validation**: Built-in data quality checks

[→ Learn about Data Pipeline](../data/pipeline.ipynb)

---

### 3.2 Data Annotation
Interactive annotation tools for creating high-quality training data:
- **Label Studio integration**: Web-based annotation interface
- **Collaborative labeling**: Multi-annotator support
- **Quality control**: Inter-annotator agreement tracking
- **Export formats**: Compatible with all training approaches

[→ Learn about Data Annotation](../data/annotation.ipynb)

---

### 3.3 Data Loading
Flexible data loading strategies for various sources:
- **HuggingFace datasets**: Direct integration with HF hub
- **Local files**: Parquet, JSON, JSONL support
- **Custom sources**: Extensible loader architecture
- **Streaming support**: Memory-efficient large dataset handling

[→ Learn about Data Loading](../data/load.ipynb)

---

### 3.4 Data Processing
Transform and prepare data for training:
- **Schema normalization**: Convert to standard format
- **Filtering and sampling**: Data selection strategies
- **Augmentation**: Expand training data diversity
- **Train/validation splits**: Automated splitting with stratification

[→ Learn about Data Processing](../data/process.ipynb)

---

## 4. Training Approaches

RM-Gallery supports three main training approaches, each optimized for different use cases:

### 4.1 Pointwise Training
**Concept**: Train models to assign absolute scores to individual responses

**Key Characteristics**:
- Scores on fixed scale (e.g., 0-4)
- Evaluates responses independently
- Suitable for rating and quality assessment
- Requires labeled scores for each response

**Use Cases**:
- Quality scoring systems
- Content moderation
- Response rating applications
- Absolute performance metrics

**Data Format**:
```json
{
  "prompt": "User query",
  "response": "Model response",
  "score": 3.5
}
```

---

### 4.2 Pairwise Training (Bradley-Terry)
**Concept**: Learn from preference comparisons between response pairs

**Key Characteristics**:
- Binary preference judgments (A > B)
- Relative comparison learning
- Bradley-Terry loss function
- Most common in RLHF workflows

**Use Cases**:
- RLHF reward models
- Preference learning
- Response ranking
- Comparative evaluation

**Data Format**:
```json
{
  "prompt": "User query",
  "chosen": "Preferred response",
  "rejected": "Less preferred response"
}
```

[→ Learn about Bradley-Terry Training](bradley_terry_rm.md)

---

### 4.3 SFT-based Training
**Concept**: Supervised fine-tuning for specialized evaluation tasks

**Key Characteristics**:
- Task-specific reward models
- Reasoning-focused evaluation
- Combines scoring with explanations
- Flexible output formats

**Use Cases**:
- Reasoning quality assessment
- Specialized domain evaluation
- Explainable scoring
- Complex evaluation criteria

**Data Format**:
```json
{
  "messages": [
    {"role": "user", "content": "Evaluation task"},
    {"role": "assistant", "content": "Analysis and score"}
  ]
}
```

[→ Learn about SFT Training](sft_rm.md)

---

## 5. Training Framework: VERL

All training approaches use the **VERL (Versatile Efficient Reinforcement Learning)** framework, which provides:

### Core Features
- **Distributed Training**: FSDP (Fully Sharded Data Parallel) support
- **Memory Efficiency**: Gradient checkpointing, mixed precision
- **Flexible Architecture**: Support for various model architectures
- **Optimization**: Advanced schedulers and optimizers
- **Logging**: WandB, SwanLab, TensorBoard integration

### Configuration Management
VERL uses Hydra for configuration:
```yaml
# trainer.yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  load_dtype: bfloat16

training:
  batch_size: 8
  learning_rate: 1e-5
  num_epochs: 3

distributed:
  strategy: fsdp
  num_gpus: 4
```

### Training Pipeline
```python
# Typical VERL training workflow
from verl import Trainer

trainer = Trainer(
    config="trainer.yaml",
    dataset=train_dataset,
    model=reward_model
)

trainer.train()
trainer.evaluate(val_dataset)
trainer.save_model("output/")
```

[→ Complete VERL Training Guide](training_rm.md)

---

## 6. Approach Comparison

### Quick Decision Guide

| Your Goal | Recommended Approach |
|-----------|---------------------|
| RLHF reward model | Pairwise (Bradley-Terry) |
| Absolute quality scoring | Pointwise |
| Reasoning evaluation | SFT-based |
| Simple preference learning | Pairwise |
| Complex evaluation criteria | SFT-based |
| Limited labeled data | Pairwise (easier annotation) |

### Detailed Comparison Matrix

| Feature | Pointwise | Pairwise (Bradley-Terry) | SFT-based |
|---------|-----------|-------------------------|-----------|
| **Data Required** | Score labels | Preference pairs | Task-specific examples |
| **Training Difficulty** | Medium | Easy | Medium-Hard |
| **Annotation Cost** | High | Low | Medium |
| **Output Type** | Absolute score | Relative preference | Flexible |
| **Best For** | Rating systems | RLHF, ranking | Specialized tasks |
| **Model Size** | Any | Any | Medium-Large |
| **Interpretability** | High | Medium | High |

## 7. Training Best Practices

### 7.1 Data Quality
- **Balanced datasets**: Ensure diverse examples across difficulty levels
- **Quality over quantity**: Better to have fewer high-quality examples
- **Regular validation**: Monitor for annotation consistency
- **Diverse sources**: Include varied scenarios and edge cases

### 7.2 Model Selection
- **Start with pretrained models**: Fine-tune instruction-tuned models
- **Size considerations**: Balance performance with inference cost
- **Architecture compatibility**: Ensure model supports your use case
- **Domain alignment**: Choose models pretrained on similar data

### 7.3 Hyperparameter Tuning
```yaml
# Recommended starting points
learning_rate: 1e-5  # Lower for larger models
batch_size: 8-32     # Based on GPU memory
epochs: 2-5          # Avoid overfitting
warmup_ratio: 0.1    # Gradual learning rate increase
weight_decay: 0.01   # Regularization
```

### 7.4 Training Monitoring
Track these key metrics:
- **Training loss**: Should decrease smoothly
- **Validation accuracy**: Primary performance indicator
- **Gradient norms**: Detect training instabilities
- **Learning rate**: Verify scheduler behavior
- **Memory usage**: Optimize for efficiency

### 7.5 Common Issues & Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Loss not decreasing | Learning rate too high | Reduce LR by 10x |
| Overfitting | Too many epochs | Reduce epochs, add regularization |
| OOM errors | Batch size too large | Reduce batch size, enable gradient checkpointing |
| Unstable training | Poor initialization | Use pretrained model, adjust warmup |
| Low accuracy | Data quality issues | Review annotations, balance dataset |

## 8. Evaluation Strategy

### During Training
- **Validation splits**: Hold out 10-20% for validation
- **Checkpoint selection**: Save based on validation performance
- **Early stopping**: Prevent overfitting
- **Multiple metrics**: Don't rely on single metric

### Post-Training
After training, evaluate your model comprehensively:

1. **Benchmark Testing**: Use standard benchmarks
   - [RewardBench](../evaluation/rewardbench2.md)
   - [JudgeBench](../evaluation/judgebench.md)
   - [RM-Bench](../evaluation/rmbench.md)

2. **Consistency Analysis**: Check for logical coherence
   - [Conflict Detector](../evaluation/conflict_detector.md)

3. **Real-world Testing**: Deploy in production-like scenarios

## 9. Complete Training Example

Here's an end-to-end example workflow:

### Step 1: Prepare Data
```python
from rm_gallery.core.data.load.base import create_loader

# Load training data
loader = create_loader(
    name="preference_data",
    load_strategy_type="huggingface",
    data_source="Anthropic/hh-rlhf",
    config={"split": "train"}
)
dataset = loader.run()
```

### Step 2: Process Data
```python
from rm_gallery.core.data.process import split_samples

# Split into train/validation
train_data, val_data = split_samples(
    dataset.datasamples,
    train_ratio=0.8
)
```

### Step 3: Configure Training
```yaml
# config/trainer.yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct

training:
  approach: bradley_terry
  batch_size: 8
  learning_rate: 1e-5
  num_epochs: 3

output:
  save_dir: ./checkpoints
  log_wandb: true
```

### Step 4: Train Model
```bash
# Run training
python examples/train/bradley-terry/trainer.py \
  --config config/trainer.yaml \
  --data_path data/train.parquet
```

### Step 5: Evaluate
```python
# Evaluate on validation set
from rm_gallery.gallery.evaluation import evaluate_reward_model

results = evaluate_reward_model(
    model_path="./checkpoints/best",
    eval_data=val_data,
    benchmark="rewardbench2"
)
```

## 10. Resource Requirements

### Hardware Recommendations

| Model Size | GPUs | Memory/GPU | Training Time* |
|------------|------|-----------|----------------|
| 1-3B | 1-2 | 16GB | 2-4 hours |
| 7-8B | 2-4 | 24GB | 4-8 hours |
| 13-14B | 4-8 | 40GB | 8-16 hours |
| 70B+ | 8+ | 80GB | 24+ hours |

*Approximate time for 10K samples, 3 epochs

### Optimization Strategies
- **Gradient checkpointing**: Reduce memory by 30-40%
- **Mixed precision (bf16/fp16)**: 2x faster training
- **FSDP**: Scale to multiple GPUs efficiently
- **Flash Attention**: 2-3x faster for long sequences
- **CPU offloading**: Train larger models with limited GPU memory

## 11. Next Steps

Ready to start training? Follow this learning path:

### For Beginners
1. **Start with data**: [Data Loading](../data/load.ipynb)
2. **Try Bradley-Terry**: [Pairwise Training](bradley_terry_rm.md)
3. **Evaluate results**: [Evaluation Overview](../evaluation/overview.md)

### For Advanced Users
1. **Custom training**: [Complete Training Guide](training_rm.md)
2. **Specialized models**: [SFT Training](sft_rm.md)
3. **Production deployment**: [RM Server](../rm_serving/rm_server.md)

### For Researchers
1. **Compare approaches**: Test all three training methods
2. **Benchmark extensively**: Use all evaluation tools
3. **Iterate and improve**: [Data Refinement](../rm_application/data_refinement.ipynb)

## 12. Additional Resources

### Documentation
- **[Data Pipeline](../data/pipeline.ipynb)** - Complete data preparation workflow
- **[Building RM](../building_rm/overview.ipynb)** - Non-training reward model construction
- **[Evaluating RM](../evaluation/overview.md)** - Comprehensive evaluation guide

### Example Code
- **[Bradley-Terry Examples](../../../examples/train/bradley-terry/)** - Complete training scripts
- **[Pairwise Examples](../../../examples/train/pairwise/)** - Pairwise training code
- **[Pointwise Examples](../../../examples/train/pointwise/)** - Pointwise training code

### Community & Support
- **GitHub Issues**: Report bugs or request features
- **Discussions**: Share experiences and ask questions
- **Contributing**: Help improve the framework

---

**Ready to train your reward model? Choose your path:**

- **[Data Pipeline](../data/pipeline.ipynb)** - Start with data preparation
- **[Complete Training Guide](training_rm.md)** - Comprehensive VERL training
- **[Bradley-Terry Training](bradley_terry_rm.md)** - Most common RLHF approach
- **[SFT Training](sft_rm.md)** - Specialized evaluation models

