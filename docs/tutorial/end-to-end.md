---
title: End-to-End Tutorial - Build Your First Reward Model
description: Complete step-by-step tutorial to build a factuality reward model from scratch in 30 minutes. Learn data preparation, model creation, evaluation, and deployment.
keywords: [tutorial, end-to-end, reward model, factuality, build, create, training, evaluation, complete guide]
tags: [tutorial, intermediate, hands-on, complete-guide]
---

# End-to-End Tutorial: Build Your First Reward Model

This tutorial will walk you through the complete process of building a reward model from scratch in under 30 minutes.

## What You'll Build

A **factuality reward model** that evaluates whether AI responses are factually accurate. This is perfect for:
- Question answering systems
- Information retrieval applications
- Content verification tools

**Time to complete**: ~20-30 minutes

---

## Prerequisites

- Python >= 3.10
- RM-Gallery installed (`pip install rm-gallery`)
- Basic Python knowledge
- (Optional) OpenAI API key for LLM-based evaluation

---

## Step 1: Setup Your Environment (2 minutes)

First, let's set up our working directory and install dependencies:

```bash
# Create project directory
mkdir my_reward_model
cd my_reward_model

# Install RM-Gallery
pip install rm-gallery

# Create a Python file
touch factuality_reward.py
```

---

## Step 2: Create Sample Data (5 minutes)

Let's create a small dataset to test our reward model. Create a file `test_data.py`:

```python
from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole

# Create test samples with factual and non-factual responses
test_samples = [
    # Sample 1: Correct fact
    DataSample(
        unique_id="fact_1",
        input=[ChatMessage(
            role=MessageRole.USER,
            content="What is the capital of France?"
        )],
        output=[
            DataOutput(answer=Step(
                role=MessageRole.ASSISTANT,
                content="The capital of France is Paris."
            )),
            DataOutput(answer=Step(
                role=MessageRole.ASSISTANT,
                content="The capital of France is London."  # Wrong!
            ))
        ]
    ),

    # Sample 2: Mathematical fact
    DataSample(
        unique_id="fact_2",
        input=[ChatMessage(
            role=MessageRole.USER,
            content="What is 15 + 27?"
        )],
        output=[
            DataOutput(answer=Step(
                role=MessageRole.ASSISTANT,
                content="15 + 27 = 42"
            )),
            DataOutput(answer=Step(
                role=MessageRole.ASSISTANT,
                content="15 + 27 = 43"  # Wrong!
            ))
        ]
    ),

    # Sample 3: Historical fact
    DataSample(
        unique_id="fact_3",
        input=[ChatMessage(
            role=MessageRole.USER,
            content="When did World War II end?"
        )],
        output=[
            DataOutput(answer=Step(
                role=MessageRole.ASSISTANT,
                content="World War II ended in 1945."
            )),
            DataOutput(answer=Step(
                role=MessageRole.ASSISTANT,
                content="World War II ended in 1950."  # Wrong!
            ))
        ]
    )
]

if __name__ == "__main__":
    print(f"Created {len(test_samples)} test samples")
    for sample in test_samples:
        print(f"  - {sample.unique_id}: {len(sample.output)} responses")
```

Run it to verify:
```bash
python test_data.py
```

**Output:**
```
Created 3 test samples
  - fact_1: 2 responses
  - fact_2: 2 responses
  - fact_3: 2 responses
```

---

## Step 3: Build a Simple Rule-Based Reward (5 minutes)

Let's start with a simple rule-based factuality checker. Create `factuality_reward.py`:

```python
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.schema import RewardResult, RewardDimensionWithScore
from rm_gallery.core.data.schema import DataSample

class SimpleFactualityReward(BasePointWiseReward):
    """A simple rule-based factuality checker."""

    name: str = "simple_factuality"

    # Define known facts (for demo purposes)
    FACTS = {
        "capital of France": "Paris",
        "15 + 27": "42",
        "World War II": "1945"
    }

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """Check if response contains correct facts."""
        response = sample.output[0].answer.content.lower()
        question = sample.input[0].content.lower()

        # Simple fact checking
        score = 0.0
        reason = "No matching facts found"

        for fact_key, fact_value in self.FACTS.items():
            if fact_key in question and fact_value.lower() in response:
                score = 1.0
                reason = f"Response contains correct fact: {fact_value}"
                break

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=score,
                    reason=reason
                )
            ]
        )

# Test it!
if __name__ == "__main__":
    from test_data import test_samples

    rm = SimpleFactualityReward()

    print("Testing Simple Factuality Reward:")
    print("=" * 60)

    for sample in test_samples:
        # Test each response
        for idx, output in enumerate(sample.output):
            test_sample = DataSample(
                unique_id=sample.unique_id,
                input=sample.input,
                output=[output]
            )
            result = rm.evaluate(test_sample)
            score = result.output[0].answer.reward.details[0].score
            reason = result.output[0].answer.reward.details[0].reason

            print(f"\n{sample.unique_id} - Response {idx + 1}:")
            print(f"  Content: {output.answer.content[:50]}...")
            print(f"  Score: {score}")
            print(f"  Reason: {reason}")
```

Run it:
```bash
python factuality_reward.py
```

**Expected Output:**
```
Testing Simple Factuality Reward:
============================================================

fact_1 - Response 1:
  Content: The capital of France is Paris....
  Score: 1.0
  Reason: Response contains correct fact: Paris

fact_1 - Response 2:
  Content: The capital of France is London....
  Score: 0.0
  Reason: No matching facts found

...
```

Great! Our simple reward model works! 🎉

---

## Step 4: Upgrade to LLM-Based Evaluation (5 minutes)

Now let's create a more sophisticated LLM-based factuality checker. This requires API credentials.

Create `llm_factuality_reward.py`:

```python
import os
from typing import Type
from pydantic import Field

from rm_gallery.core.reward.base import BaseLLMReward, BasePointWiseReward
from rm_gallery.core.reward.schema import RewardResult, RewardDimensionWithScore
from rm_gallery.core.reward.template import BasePromptTemplate
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.utils.prompt import format_messages

# Define evaluation template
class FactualityTemplate(BasePromptTemplate):
    factual: bool = Field(
        default=...,
        description="Is the response factually accurate? true or false"
    )
    confidence: float = Field(
        default=...,
        description="Confidence score from 0.0 to 1.0"
    )
    reason: str = Field(
        default=...,
        description="Brief explanation of the factuality assessment"
    )

    @classmethod
    def format(cls, question: str, answer: str, **kwargs) -> str:
        return f"""Evaluate the factual accuracy of the following response.

Question: {question}

Response: {answer}

Instructions:
1. Determine if the response is factually accurate
2. Provide a confidence score (0.0 = completely wrong, 1.0 = completely correct)
3. Explain your reasoning

# Output Format:
{cls.schema()}
"""

# Define LLM-based reward
class LLMFactualityReward(BaseLLMReward, BasePointWiseReward):
    """LLM-based factuality assessment reward module."""

    name: str = "llm_factuality"
    template: Type[BasePromptTemplate] = FactualityTemplate

    def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
        """Prepare prompt parameters."""
        question = format_messages(sample.input)
        answer = sample.output[0].answer.content
        return {"question": question, "answer": answer}

    def _after_evaluate(self, response: FactualityTemplate, **kwargs) -> RewardResult:
        """Parse LLM response into reward value."""
        score = response.confidence if response.factual else 0.0

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=score,
                    reason=f"Factual: {response.factual} ({response.reason})"
                )
            ],
            extra_data={
                "factual": response.factual,
                "confidence": response.confidence
            }
        )

# Test it!
if __name__ == "__main__":
    # Set up API credentials
    os.environ["OPENAI_API_KEY"] = "your_api_key_here"
    os.environ["BASE_URL"] = "your_base_url_here"  # Optional

    # Initialize LLM
    llm = OpenaiLLM(model="gpt-4", enable_thinking=True)

    # Create reward model
    rm = LLMFactualityReward(llm=llm)

    from test_data import test_samples

    print("Testing LLM Factuality Reward:")
    print("=" * 60)

    for sample in test_samples[:1]:  # Test first sample only
        # Evaluate both responses
        for idx, output in enumerate(sample.output):
            test_sample = DataSample(
                unique_id=sample.unique_id,
                input=sample.input,
                output=[output]
            )
            result = rm.evaluate(test_sample)
            score = result.output[0].answer.reward.details[0].score
            reason = result.output[0].answer.reward.details[0].reason

            print(f"\n{sample.unique_id} - Response {idx + 1}:")
            print(f"  Content: {output.answer.content}")
            print(f"  Score: {score:.2f}")
            print(f"  Reason: {reason}")
```

---

## Step 5: Compare Multiple Responses (5 minutes)

Now let's upgrade to listwise evaluation to compare responses side-by-side:

```python
from rm_gallery.core.reward.base import BaseListWiseReward
from rm_gallery.core.reward.schema import RewardDimensionWithRank
from typing import List

class ComparativeFactualityReward(BaseLLMReward, BaseListWiseReward):
    """Compare factual accuracy of multiple responses."""

    name: str = "comparative_factuality"
    template: Type[BasePromptTemplate] = ComparativeTemplate

    def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
        question = format_messages(sample.input)
        responses = [output.answer.content for output in sample.output]
        return {"question": question, "responses": responses}

    def _after_evaluate(self, response: ComparativeTemplate, **kwargs) -> RewardResult:
        # Convert ranking to scores
        num_responses = len(response.rankings)
        scores = [0.0] * num_responses

        for rank_info in response.rankings:
            idx = rank_info["response_index"] - 1
            rank = rank_info["rank"]
            # Higher rank = higher score
            scores[idx] = 1.0 - (rank - 1) / (num_responses - 1) if num_responses > 1 else 1.0

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name,
                    reason=response.explanation,
                    rank=scores
                )
            ]
        )
```

---

## Step 6: Register Your Reward Model (3 minutes)

Make your reward model easily accessible:

```python
# In factuality_reward.py, add:

from rm_gallery.core.reward.registry import RewardRegistry

# Register your reward models
RewardRegistry.register("simple_factuality", SimpleFactualityReward)
RewardRegistry.register("llm_factuality", LLMFactualityReward)

# Now you can use them like pre-built models:
# rm = RewardRegistry.get("simple_factuality")
```

---

## Step 7: Batch Evaluation (5 minutes)

Let's evaluate all samples efficiently:

```python
# evaluate_all.py
import os
from test_data import test_samples
from factuality_reward import SimpleFactualityReward

# Initialize reward model
rm = SimpleFactualityReward()

# Batch evaluate
print("Running batch evaluation...")
results = rm.evaluate_batch(
    test_samples,
    max_workers=4  # Parallel processing
)

# Calculate accuracy
correct = 0
total = 0

for sample in results:
    for output in sample.output:
        if output.answer.reward:
            score = output.answer.reward.details[0].score
            # Assume first response is correct, second is wrong
            is_first = sample.output.index(output) == 0
            if (is_first and score > 0.5) or (not is_first and score <= 0.5):
                correct += 1
            total += 1

accuracy = correct / total if total > 0 else 0
print(f"\n📊 Evaluation Results:")
print(f"  Total responses: {total}")
print(f"  Correct predictions: {correct}")
print(f"  Accuracy: {accuracy:.1%}")
```

Run it:
```bash
python evaluate_all.py
```

---

## 🎉 Congratulations!

You've successfully built a complete reward model from scratch! Here's what you accomplished:

✅ Created test data
✅ Built a rule-based reward model
✅ Upgraded to LLM-based evaluation
✅ Implemented listwise comparison
✅ Registered your model
✅ Ran batch evaluations

## Next Steps

### 1. Improve Your Model

- Add more sophisticated fact-checking logic
- Integrate external knowledge bases
- Fine-tune prompts for better accuracy

### 2. Train Your Own Model

Instead of using rule-based or LLM-based rewards, train a dedicated model:

```bash
cd examples/train/pointwise
./run_pointwise.sh
```

See the [Training Guide](training_rm/training_rm.md)

### 3. Deploy to Production

Deploy your reward model as a service:

```bash
# See the RM Serving Guide
```

See the [RM Server Guide](rm_serving/rm_server.md)

### 4. Evaluate on Benchmarks

Test your model on standard benchmarks:

- [RewardBench2](evaluation/rewardbench2.md)
- [RM-Bench](evaluation/rmbench.md)
- [Conflict Detector](evaluation/conflict_detector.md)

### 5. Share Your Work

Consider contributing your reward model to RM-Gallery:

1. Fork the repository
2. Add your model to `rm_gallery/gallery/rm/`
3. Submit a pull request

See our [Contribution Guide](../../contribution.md)

---

## Troubleshooting

**Problem**: Import errors
```bash
# Solution: Reinstall RM-Gallery
pip install --upgrade rm-gallery
```

**Problem**: API errors
```python
# Solution: Check credentials
import os
print(os.environ.get("OPENAI_API_KEY"))
```

**Problem**: Slow evaluation
```python
# Solution: Use batch processing
results = rm.evaluate_batch(samples, max_workers=8)
```

---

## Complete Code

Find the complete code for this tutorial in:
- `examples/end_to_end/` (coming soon)

Or download it:
```bash
git clone https://github.com/modelscope/RM-Gallery.git
cd RM-Gallery/examples/end_to_end/
```

---

## Additional Resources

- 📚 [Full Documentation](../index.md)
- 💻 [Interactive Notebooks](../../examples/)
- 🤝 [Community Forum](https://github.com/modelscope/RM-Gallery/discussions)
- 📝 [API Reference](../api_reference.md)

Happy building! 🚀

