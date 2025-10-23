---
title: Frequently Asked Questions (FAQ)
description: Find answers to common questions about RM-Gallery, including installation, building reward models, training, evaluation, and troubleshooting.
keywords: [FAQ, questions, troubleshooting, help, guide, reward model, RM-Gallery, problems, solutions]
tags: [faq, help, troubleshooting, guide]
---

# Frequently Asked Questions (FAQ)

Find answers to common questions about RM-Gallery.

## General Questions

### What is RM-Gallery?

RM-Gallery is a comprehensive platform for training, building, and applying reward models. It provides:

- 35+ pre-built reward models for various scenarios
- Unified architecture for custom reward model development
- Integration with training frameworks (VERL)
- Benchmark evaluation tools
- Production-ready serving capabilities

### Who should use RM-Gallery?

RM-Gallery is designed for:

- **Researchers** evaluating reward models on benchmarks
- **ML Engineers** building and deploying reward models in production
- **AI Practitioners** training reward models for RLHF/post-training
- **Developers** integrating reward models into applications

### What scenarios does RM-Gallery support?

RM-Gallery covers diverse evaluation scenarios:

- **Math**: Correctness verification, step-by-step reasoning
- **Code**: Quality assessment, syntax checking, execution correctness
- **Alignment**: Helpfulness, harmlessness, honesty (3H)
- **General**: Accuracy, F1, ROUGE, factuality
- **Format & Style**: Length, repetition, privacy compliance

---

## Getting Started

### How do I install RM-Gallery?

```bash
# From PyPI (recommended)
pip install rm-gallery

# From source
git clone https://github.com/modelscope/RM-Gallery.git
cd RM-Gallery
pip install .
```

**Requirements**: Python >= 3.10 and < 3.13

### Do I need API keys to use RM-Gallery?

It depends on which reward models you use:

- **Rule-based models** (e.g., length checks, format validation): No API key needed
- **LLM-based models** (e.g., helpfulness, safety evaluation): API key required

For LLM-based models, set up your credentials:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key"
os.environ["BASE_URL"] = "your_base_url"  # Optional: custom endpoint
```

### How do I choose the right reward model?

1. **Check available models**: `RewardRegistry.list()`
2. **Match your scenario**: See the [RM Library](library/rm_library.md)
3. **Consider complexity**:
   - Simple tasks → Rule-based models
   - Complex evaluation → LLM-based models
4. **Test on your data**: Try multiple models and compare results

---

## Building Reward Models

### Can I create my own reward model?

Yes! RM-Gallery provides multiple levels of customization:

**Level 1: Custom Rubrics** (Easiest)
```python
from rm_gallery.gallery.rm.alignment.base import BaseHarmlessnessListWiseReward

custom_rm = BaseHarmlessnessListWiseReward(
    name="custom_safety",
    rubrics=["Your criterion 1", "Your criterion 2"],
    llm=llm
)
```

**Level 2: Custom LLM Template**
- Inherit from `BaseLLMReward`
- Override `_before_evaluate` and `_after_evaluate`
- See [Custom RM Tutorial](tutorial/building_rm/custom_reward.md)

**Level 3: Full Custom Logic**
- Inherit from `BasePointWiseReward` or `BaseListWiseReward`
- Implement `_evaluate` method
- Complete control over evaluation logic

### What's the difference between pointwise, pairwise, and listwise?

- **Pointwise**: Evaluates each response independently (e.g., "Is this response factually correct?")
  - Use case: Grammar checking, length validation, format compliance

- **Pairwise**: Compares two responses directly (e.g., "Which response is better?")
  - Use case: Preference learning, A/B testing

- **Listwise**: Ranks multiple responses (e.g., "Rank these 5 responses from best to worst")
  - Use case: Best-of-N selection, multi-candidate evaluation

### How do I use the Rubric-Critic-Score paradigm?

The Rubric-Critic-Score paradigm follows three steps:

1. **Rubric**: Define evaluation criteria
2. **Critic**: LLM analyzes responses based on rubrics
3. **Score**: Get numerical scores

```python
# 1. Define rubrics
rubrics = [
    "Response must be factually accurate",
    "Response should be concise and clear",
    "Response must be helpful to the user"
]

# 2. Create rubric-based reward
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

rm = BaseHelpfulnessListWiseReward(
    name="helpful_rm",
    rubrics=rubrics,
    llm=llm
)

# 3. Evaluate and get scores
result = rm.evaluate(sample)
```

---

## Training

### How do I train my own reward model?

RM-Gallery integrates with the VERL framework:

```bash
# 1. Prepare training data
python examples/data/data_from_yaml.py --config config.yaml

# 2. Launch distributed training
cd examples/train/pointwise
./run_pointwise.sh
```

See the [Training Guide](tutorial/training_rm/training_rm.md) for details.

### What training paradigms are supported?

- **Pointwise**: Train models to assign absolute scores
- **Pairwise**: Train models to predict preferences between pairs
- **Bradley-Terry**: Train models using the Bradley-Terry model

### Do I need multiple GPUs for training?

It depends on model size and dataset:

- **Small models (<3B)**: Single GPU possible
- **Medium models (3B-14B)**: 2-4 GPUs recommended
- **Large models (>14B)**: 8+ GPUs recommended

RM-Gallery supports distributed training via Ray.

---

## Evaluation

### What benchmarks are available?

RM-Gallery supports multiple standard benchmarks:

- **[RewardBench2](tutorial/evaluation/rewardbench2.md)**: Latest comprehensive benchmark
- **[RM-Bench](tutorial/evaluation/rmbench.md)**: Multi-dimensional evaluation
- **[Conflict Detector](tutorial/evaluation/conflict_detector.md)**: Identify evaluation conflicts
- **[JudgeBench](tutorial/evaluation/judgebench.md)**: Judge capability testing

### How do I run evaluations efficiently?

Use batch processing with parallel workers:

```python
# Batch evaluation (recommended)
results = rm.evaluate_batch(
    samples,
    max_workers=8  # Adjust based on your resources
)

# vs. Sequential evaluation (slower)
results = [rm.evaluate(sample) for sample in samples]
```

Batch processing can be **10-100x faster** for large datasets!

### How do I interpret reward scores?

Score interpretation depends on the reward model:

- **Binary scores**: 0.0 (bad) to 1.0 (good)
- **Continuous scores**: Usually normalized to [0, 1]
- **Relative scores**: Compare across responses (e.g., ranking)

Always check the `reason` field for detailed explanations:

```python
for detail in result.output[0].answer.reward.details:
    print(f"Score: {detail.score}")
    print(f"Reason: {detail.reason}")
```

---

## Production & Deployment

### How do I deploy reward models in production?

RM-Gallery supports high-performance serving via New API:

1. **Set up New API server** (see [RM Server Guide](tutorial/rm_serving/rm_server.md))
2. **Deploy your reward model**
3. **Update client code**:

```python
os.environ["BASE_URL"] = "https://your-api-endpoint.com"
rm = RewardRegistry.get("your_reward_model")
```

Benefits:
- High throughput (100+ requests/sec)
- Fault tolerance
- Unified API management
- Easy scaling

### Can I use RM-Gallery with RLHF?

Yes! RM-Gallery integrates seamlessly with RLHF pipelines:

```python
# Use reward model as RLHF reward function
from rm_gallery.core.reward.registry import RewardRegistry

reward_fn = RewardRegistry.get("alignment_reward")

# Integrate with your RLHF framework
# See examples/train/rl_training/
```

### How do I handle rate limits?

For API-based reward models:

1. **Batch processing**: Reduce number of API calls
2. **Parallel workers**: Control `max_workers` parameter
3. **Caching**: Cache results for duplicate queries
4. **Local deployment**: Use New API for unlimited throughput

---

## Troubleshooting

### Import errors: "No module named 'rm_gallery'"

**Solution**: Make sure RM-Gallery is installed:
```bash
pip install rm-gallery
# or
pip install -e .  # if installing from source
```

### API errors: "Authentication failed"

**Solution**: Check your API credentials:
```python
import os
print(os.environ.get("OPENAI_API_KEY"))  # Should not be None
print(os.environ.get("BASE_URL"))
```

### Evaluation is too slow

**Solutions**:
1. Use `evaluate_batch()` instead of individual `evaluate()` calls
2. Increase `max_workers` (but respect API rate limits)
3. Use rule-based models instead of LLM-based when possible
4. Deploy local serving with New API

### "KeyError: reward model not found"

**Solution**: Check available models:
```python
from rm_gallery.core.reward.registry import RewardRegistry
print(RewardRegistry.list())  # See all available models
```

### Memory errors during training

**Solutions**:
1. Reduce batch size in training config
2. Use gradient accumulation
3. Enable mixed precision training (fp16/bf16)
4. Use more GPUs with data parallelism

---

## Contributing

### How can I contribute to RM-Gallery?

We welcome contributions! You can:

1. **Add new reward models** to the gallery
2. **Improve documentation** and examples
3. **Report bugs** via GitHub Issues
4. **Submit benchmarks** for evaluation
5. **Share use cases** and best practices

See our [Contribution Guide](contribution.md) for details.

### Can I contribute a benchmark?

Yes! We encourage benchmark contributions:

1. Prepare your dataset in JSONL format
2. Create a data loader (see `rm_gallery/gallery/data/`)
3. Submit a PR with documentation
4. We'll review and integrate it

### How do I report a bug?

1. Check if it's already reported: [GitHub Issues](https://github.com/modelscope/RM-Gallery/issues)
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Environment info (Python version, RM-Gallery version)
   - Error messages/logs

---

## Best Practices

### What are the best practices for building reward models?

1. **Start with pre-built models**: Test existing models before building custom ones
2. **Use rubrics**: Define clear evaluation criteria
3. **Validate on diverse data**: Test on multiple datasets
4. **Iterate**: Refine based on evaluation results
5. **Document**: Keep track of model versions and performance

### How do I ensure reward model quality?

1. **Human evaluation**: Compare against human judgments
2. **Cross-validation**: Test on held-out data
3. **Benchmark testing**: Evaluate on standard benchmarks
4. **A/B testing**: Compare multiple reward models
5. **Monitor in production**: Track performance metrics

### What data format should I use?

RM-Gallery uses a standardized data schema:

```python
from rm_gallery.core.data.schema import DataSample, DataOutput, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole

sample = DataSample(
    unique_id="example_1",
    input=[ChatMessage(role=MessageRole.USER, content="...")],
    output=[DataOutput(answer=Step(role=MessageRole.ASSISTANT, content="..."))]
)
```

See [Data Pipeline Tutorial](tutorial/data/pipeline.md) for details.

---

## Still Have Questions?

- 📚 **[Full Documentation](index.md)** - Comprehensive guides
- 💬 **[GitHub Discussions](https://github.com/modelscope/RM-Gallery/discussions)** - Community Q&A
- 🐛 **[Report Issues](https://github.com/modelscope/RM-Gallery/issues)** - Bug reports
- 🤝 **[Contribution Guide](contribution.md)** - Get involved

---

**Last Updated**: October 2025

