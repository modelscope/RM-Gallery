# Reward Model Gallery

Welcome to the RM-Gallery! This directory contains 35+ pre-built reward models ready to use for various evaluation scenarios.

## 🎨 Gallery Overview

```
rm/
├── math/           # Mathematical correctness & reasoning
├── code/           # Code quality & execution
├── alignment/      # Helpfulness, harmlessness, honesty (3H)
├── general/        # General-purpose metrics
└── format/         # Format & style checking
```

## 📚 Available Scenarios

### 🔢 Math (`math/`)

Reward models for mathematical tasks:

| Model | Description | Use Case |
|-------|-------------|----------|
| `math_correctness_reward` | Verify mathematical correctness | Math QA, problem solving |
| `math_step_reward` | Evaluate reasoning steps | Step-by-step math solutions |
| `equation_reward` | Check equation validity | Algebraic problem solving |

**Example Usage:**
```python
from rm_gallery.core.reward.registry import RewardRegistry

rm = RewardRegistry.get("math_correctness_reward")
result = rm.evaluate(sample)
```

### 💻 Code (`code/`)

Reward models for code evaluation:

| Model | Description | Use Case |
|-------|-------------|----------|
| `code_quality_reward` | Assess code quality | Code review, refactoring |
| `code_syntax_reward` | Check syntax correctness | Syntax validation |
| `code_execution_reward` | Verify execution correctness | Unit testing, validation |
| `code_style_reward` | Check style compliance | Code formatting, linting |

**Example Usage:**
```python
rm = RewardRegistry.get("code_quality_reward")
result = rm.evaluate(sample)
```

### 🤝 Alignment (`alignment/`)

Reward models for AI alignment (3H: Helpful, Harmless, Honest):

| Model | Description | Use Case |
|-------|-------------|----------|
| `helpfulness_listwise_reward` | Evaluate helpfulness | Ranking helpful responses |
| `harmlessness_listwise_reward` | Assess safety/harmlessness | Safety evaluation |
| `honesty_listwise_reward` | Verify honesty/truthfulness | Factuality checking |
| `safety_listwise_reward` | General safety check | Content moderation |

**Example Usage:**
```python
rm = RewardRegistry.get("helpfulness_listwise_reward")
result = rm.evaluate(sample)
```

### 🎯 General (`general/`)

General-purpose evaluation metrics:

| Model | Description | Use Case |
|-------|-------------|----------|
| `accuracy_reward` | Calculate accuracy | Classification tasks |
| `f1_reward` | F1 score calculation | Binary/multi-class eval |
| `rouge_reward` | ROUGE metrics | Text summarization |
| `bleu_reward` | BLEU score | Machine translation |

**Example Usage:**
```python
rm = RewardRegistry.get("accuracy_reward")
result = rm.evaluate(sample)
```

### 📝 Format (`format/`)

Format and style checking:

| Model | Description | Use Case |
|-------|-------------|----------|
| `length_reward` | Check response length | Length constraints |
| `format_reward` | Verify format compliance | Structured output |
| `repetition_reward` | Detect repetition | Quality control |
| `privacy_reward` | Check PII exposure | Privacy compliance |

**Example Usage:**
```python
rm = RewardRegistry.get("length_reward")
result = rm.evaluate(sample)
```

## 🚀 Quick Start

### 1. List All Available Models

```python
from rm_gallery.core.reward.registry import RewardRegistry

# List all models
all_models = RewardRegistry.list()
print(f"Total models: {len(all_models)}")

# Filter by category
math_models = [m for m in all_models if "math" in m]
print(f"Math models: {math_models}")
```

### 2. Load and Use a Model

```python
# Load a pre-built model
rm = RewardRegistry.get("helpfulness_listwise_reward")

# Prepare data
from rm_gallery.core.data.schema import DataSample, DataOutput, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole

sample = DataSample(
    unique_id="example",
    input=[ChatMessage(role=MessageRole.USER, content="What is AI?")],
    output=[
        DataOutput(answer=Step(
            role=MessageRole.ASSISTANT,
            content="AI is artificial intelligence..."
        ))
    ]
)

# Evaluate
result = rm.evaluate(sample)
score = result.output[0].answer.reward.details[0].score
print(f"Score: {score}")
```

### 3. Batch Evaluation

```python
# Evaluate multiple samples efficiently
results = rm.evaluate_batch(
    samples,
    max_workers=8  # Parallel processing
)
```

## 🛠️ Customization

### Use with Custom Rubrics

Many alignment models support custom rubrics:

```python
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward
from rm_gallery.core.model.openai_llm import OpenaiLLM

# Set up LLM
llm = OpenaiLLM(model="gpt-4", enable_thinking=True)

# Create custom helpfulness evaluator
custom_rm = BaseHelpfulnessListWiseReward(
    name="custom_helpful",
    desc="Custom helpfulness evaluation",
    scenario="Customer support",
    rubrics=[
        "Response directly addresses the question",
        "Response is clear and easy to understand",
        "Response provides actionable information"
    ],
    llm=llm
)

result = custom_rm.evaluate(sample)
```

### Combine Multiple Rewards

```python
from rm_gallery.core.reward.composition import RewardComposition

# Combine multiple reward models
composition = RewardComposition(
    rewards=[
        RewardRegistry.get("helpfulness_listwise_reward"),
        RewardRegistry.get("harmlessness_listwise_reward"),
        RewardRegistry.get("honesty_listwise_reward")
    ],
    weights=[0.4, 0.4, 0.2]  # Weighted combination
)

result = composition.evaluate(sample)
```

## 📖 Model Documentation

Each reward model has detailed documentation:

### Checking Model Details

```python
rm = RewardRegistry.get("helpfulness_listwise_reward")

# Get model info
print(f"Name: {rm.name}")
print(f"Description: {rm.desc if hasattr(rm, 'desc') else 'N/A'}")
print(f"Type: {type(rm).__name__}")
```

### Model Categories

- **Rule-based**: Fast, deterministic, no API needed
- **LLM-based**: Sophisticated, requires API credentials
- **Hybrid**: Combines rule-based and LLM evaluation

## 🧪 Testing Models

Test models on your data:

```python
# Quick test
def test_reward_model(rm, test_samples):
    """Test a reward model on samples."""
    results = rm.evaluate_batch(test_samples)

    scores = []
    for result in results:
        for output in result.output:
            if output.answer.reward:
                score = output.answer.reward.details[0].score
                scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average score: {avg_score:.2f}")
    print(f"Score range: [{min(scores):.2f}, {max(scores):.2f}]")

# Run test
test_reward_model(rm, my_test_samples)
```

## 🔧 Configuration

### API Setup (for LLM-based models)

```python
import os

# Set API credentials
os.environ["OPENAI_API_KEY"] = "your_api_key"
os.environ["BASE_URL"] = "your_base_url"  # Optional

# Initialize model
rm = RewardRegistry.get("helpfulness_listwise_reward")
```

### Model Parameters

Many models accept parameters:

```python
from rm_gallery.gallery.rm.format.length import LengthReward

# Create with custom parameters
rm = LengthReward(
    name="custom_length",
    min_length=50,
    max_length=200,
    penalty_factor=0.1
)
```

## 📊 Benchmarking

Compare models on benchmarks:

```python
from rm_gallery.core.data.load.base import create_loader

# Load benchmark
loader = create_loader(
    name="rewardbench2",
    load_strategy_type="huggingface",
    data_source="rewardbench2"
)
dataset = loader.run()

# Evaluate multiple models
models = [
    "helpfulness_listwise_reward",
    "harmlessness_listwise_reward",
    "honesty_listwise_reward"
]

for model_name in models:
    rm = RewardRegistry.get(model_name)
    results = rm.evaluate_batch(dataset.datasamples[:100])
    # Calculate metrics...
    print(f"{model_name}: accuracy = {accuracy:.2%}")
```

## 🤝 Contributing New Models

Want to add your reward model to the gallery?

1. **Create your model** in the appropriate category
2. **Register it** in `__init__.py`
3. **Add tests** in `tests/rm/`
4. **Submit a PR** with documentation

See [Contribution Guide](../../../docs/contribution.md)

### Example Contribution

```python
# rm_gallery/gallery/rm/custom/my_reward.py
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.schema import RewardResult, RewardDimensionWithScore

class MyReward(BasePointWiseReward):
    """My custom reward model."""

    name: str = "my_reward"

    def _evaluate(self, sample, **kwargs) -> RewardResult:
        # Your logic here
        return RewardResult(...)

# rm_gallery/gallery/rm/__init__.py
from .custom.my_reward import MyReward
from rm_gallery.core.reward.registry import RewardRegistry

RewardRegistry.register("my_reward", MyReward)
```

## 📚 Additional Resources

- [Building Custom RM](../../../docs/tutorial/building_rm/custom_reward.md)
- [RM Library Documentation](../../../docs/library/rm_library.md)
- [API Reference](../../../docs/api_reference.md)
- [Examples](../../../examples/)

## 💡 Tips

1. **Start simple**: Try pre-built models before building custom ones
2. **Test thoroughly**: Validate on diverse datasets
3. **Batch processing**: Use `evaluate_batch()` for efficiency
4. **Combine models**: Use RewardComposition for multi-aspect evaluation
5. **Monitor performance**: Track scores and adjust as needed

---

**Questions?** Check our [FAQ](../../../docs/faq.md) or ask in [GitHub Discussions](https://github.com/modelscope/RM-Gallery/discussions)

