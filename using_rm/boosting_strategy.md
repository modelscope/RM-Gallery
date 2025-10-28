<!-- # Boosting Strategy

## Overview

Boosting strategies leverage reward models to improve the quality of LLM outputs. Instead of relying on a single generation, these strategies generate multiple candidates and use reward models to select or combine the best ones.

## Best-of-N Selection

Best-of-N is a simple yet effective strategy that generates multiple responses and selects the top one(s) based on reward scores.

### How It Works

1. Generate `N` candidate responses for a given prompt
2. Score each response using a reward model
3. Select the response(s) with the highest score(s)

### Quick Example

```python
from rm_gallery.core.data.schema import DataSample, DataOutput, Step
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.registry import RewardRegistry

# Initialize LLM and reward model
llm = OpenaiLLM(model="qwen3-8b")
reward = RewardRegistry.get("base_helpfulness_listwise")(
    name="helpfulness",
    llm=llm,
    rubrics=["Judge according to your own standard"]
)

# Create sample with multiple generated responses
sample = DataSample(
    unique_id="demo",
    input=[ChatMessage(role="user", content="Explain why exercise is important.")],
    output=[
        DataOutput(answer=Step(content="Response 1...")),
        DataOutput(answer=Step(content="Response 2...")),
        DataOutput(answer=Step(content="Response 3...")),
        # ... more responses
    ]
)

# Select the best response
best_sample = reward.best_of_n(sample=sample, n=1)
print(f"Best score: {best_sample.output[0].answer.reward.score:.2f}")
print(f"Best response: {best_sample.output[0].answer.content}")
```

### Use Cases

- **Content Generation**: Select the most coherent or creative text
- **Code Generation**: Choose the most correct or efficient solution
- **Question Answering**: Pick the most accurate and helpful answer
- **Translation**: Select the most natural translation

### Considerations

- **Cost vs Quality**: More candidates (higher N) typically improve quality but increase computational cost
- **Diversity**: Ensure generated candidates are sufficiently diverse (use temperature, prompt variations, etc.)
- **Reward Model Selection**: Choose a reward model appropriate for your task (helpfulness, harmlessness, factuality, etc.)

## Other Strategies

### Model Ensemble
!!! info "Coming Soon"
    Combine multiple reward models to leverage their complementary strengths.

### Weighted Combination
!!! info "Coming Soon"
    Generate multiple responses and combine them based on weighted scores.

### Cascading Models
!!! info "Coming Soon"
    Use a pipeline of reward models with increasing complexity.

### Adaptive Boosting
!!! info "Coming Soon"
    Dynamically adjust generation strategies based on reward feedback.

## Related Topics

For more applications of reward models, see:
- [Post Training with RM](../tutorial/rm_application/post_training.md)
- [Data Refinement](../tutorial/rm_application/data_refinement.md)

-->
