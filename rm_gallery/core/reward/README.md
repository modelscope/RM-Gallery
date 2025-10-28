# Reward Module Architecture

This directory contains the core reward model architecture and base classes for RM-Gallery.

## 📁 Directory Structure

```
reward/
├── base.py              # Base reward classes (BaseReward, BasePointWiseReward, etc.)
├── schema.py            # Reward result schemas (RewardResult, RewardDimension, etc.)
├── template.py          # Prompt template base classes
├── registry.py          # Reward model registry for easy access
└── rubric/
    ├── generator.py     # Auto-rubric generation
    ├── structurer.py    # Rubric structuring
    └── analyzer.py      # Rubric analysis
```

## 🏗️ Core Architecture

### Base Classes

```
BaseReward (Abstract)
├── BasePointWiseReward      # Evaluate individual responses
├── BaseListWiseReward       # Compare multiple responses
│   └── BasePairWiseReward   # Compare two responses (specialized)
├── BaseStepWiseReward       # Evaluate reasoning steps
└── BaseLLMReward            # LLM-based evaluation
    └── BaseRubricReward     # Rubric-guided LLM evaluation
```

### When to Use Each

**BasePointWiseReward**
- Evaluate each response independently
- Examples: Grammar check, length validation, format compliance
- Use case: "Is this response correct?"

**BaseListWiseReward**
- Compare and rank multiple responses
- Examples: Best-of-N selection, preference ranking
- Use case: "Which response is best?"

**BasePairWiseReward**
- Compare two responses directly
- Examples: A/B testing, preference learning
- Use case: "Which is better: A or B?"

**BaseStepWiseReward**
- Evaluate multi-step reasoning
- Examples: Math problem solving, chain-of-thought
- Use case: "Are the reasoning steps correct?"

**BaseLLMReward**
- Use LLM for sophisticated evaluation
- Examples: Factuality, helpfulness, safety
- Use case: "Is this response helpful?"

**BaseRubricReward**
- LLM evaluation guided by rubrics
- Examples: Essay grading, quality assessment
- Use case: "Rate this response on specific criteria"

## 📝 Creating Custom Rewards

### Method 1: Rule-Based Reward

```python
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.schema import RewardResult, RewardDimensionWithScore
from rm_gallery.core.data.schema import DataSample

class CustomReward(BasePointWiseReward):
    name: str = "custom_reward"

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        # Your evaluation logic here
        response = sample.output[0].answer.content
        score = your_scoring_function(response)

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=score,
                    reason="Your explanation"
                )
            ]
        )
```

### Method 2: LLM-Based Reward

```python
from rm_gallery.core.reward.base import BaseLLMReward, BasePointWiseReward
from rm_gallery.core.reward.template import BasePromptTemplate
from pydantic import Field

class CustomTemplate(BasePromptTemplate):
    score: float = Field(default=..., description="Score from 0 to 1")
    reason: str = Field(default=..., description="Explanation")

    @classmethod
    def format(cls, question: str, answer: str, **kwargs) -> str:
        return f"""
Question: {question}
Answer: {answer}

Evaluate this answer and provide:
{cls.schema()}
"""

class CustomLLMReward(BaseLLMReward, BasePointWiseReward):
    name: str = "custom_llm_reward"
    template: Type[BasePromptTemplate] = CustomTemplate

    def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
        return {
            "question": format_messages(sample.input),
            "answer": sample.output[0].answer.content
        }

    def _after_evaluate(self, response: CustomTemplate, **kwargs) -> RewardResult:
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=response.score,
                    reason=response.reason
                )
            ]
        )
```

### Method 3: Rubric-Based Reward

```python
from rm_gallery.gallery.rm.alignment.base import BaseHarmlessnessListWiseReward

reward = BaseHarmlessnessListWiseReward(
    name="custom_safety",
    desc="Safety evaluation",
    scenario="AI Assistant",
    rubrics=[
        "Response must not provide harmful information",
        "Response should redirect appropriately",
        "Response must follow ethical guidelines"
    ],
    llm=llm
)
```

## 🔧 Key Components

### RewardResult

The evaluation result structure:

```python
class RewardResult:
    name: str                           # Reward model name
    details: List[RewardDimension]      # Detailed scores/ranks
    extra_data: Optional[dict]          # Additional data
```

### RewardDimension Types

- **RewardDimensionWithScore**: For pointwise evaluation (single score)
- **RewardDimensionWithRank**: For listwise evaluation (ranking)

### Template System

All LLM-based rewards use the template system:

1. **Define schema**: Specify expected LLM output format
2. **Format prompt**: Create evaluation prompt
3. **Parse response**: Extract structured data from LLM

## 📊 Registry Pattern

Register rewards for easy access:

```python
from rm_gallery.core.reward.registry import RewardRegistry

# Register
RewardRegistry.register("my_reward", MyRewardClass)

# Use
rm = RewardRegistry.get("my_reward")
result = rm.evaluate(sample)
```

## 🧪 Testing

Test your reward models:

```python
# Create test sample
from rm_gallery.core.data.schema import DataSample, DataOutput, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole

sample = DataSample(
    unique_id="test",
    input=[ChatMessage(role=MessageRole.USER, content="Test question")],
    output=[DataOutput(answer=Step(
        role=MessageRole.ASSISTANT,
        content="Test answer"
    ))]
)

# Evaluate
rm = MyReward()
result = rm.evaluate(sample)

# Check result
assert result.output[0].answer.reward is not None
score = result.output[0].answer.reward.details[0].score
print(f"Score: {score}")
```

## 📚 Additional Resources

- [Custom Reward Tutorial](../../../docs/tutorial/building_rm/custom_reward.md)
- [Reward Model Gallery](../../gallery/rm/)
- [API Reference](../../../docs/api_reference.md)

## 🤝 Contributing

To add a new reward model:

1. Inherit from appropriate base class
2. Implement required methods
3. Add tests
4. Submit PR

See [Contribution Guide](../../../docs/contribution.md)

