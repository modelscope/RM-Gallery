# Best-of-N Selection with LLM-based Reward Models

## 1. Overview
This tutorial demonstrates how to implement a Best-of-N selection system using LLM-based reward models. The system generates multiple responses to a given prompt and selects the best one based on reward scores.

Key Concepts:

- **Best-of-N**: Generates multiple responses and selects the top one based on reward scores

- **Reward Model**: Evaluates response quality using rubrics like helpfulness, harmlessness, etc.

- **LLM Integration**: Uses LLMs for both response generation and reward scoring

## 2. Setup

First, let's import necessary modules:


```python
# Import core modules
import sys
sys.path.append('../../..')

from concurrent.futures import ThreadPoolExecutor
from rm_gallery.core.data.schema import DataSample, DataOutput, Step
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.registry import RewardRegistry
import numpy as np
```

## 3. Create Sample Input

Let's start by creating a sample input to work with.


```python
# Create a sample input
sample = DataSample(
    unique_id="best_of_n_demo",
    input=[
        ChatMessage(
            role="user",
            content="Explain why maintaining a balanced diet is important for health."
        )
    ],
    output=[],  # We'll generate responses later
)
```

## 4. Generate Multiple Responses

We'll use an LLM to generate multiple candidate responses.


```python
# Initialize LLM for response generation
llm = OpenaiLLM(model="qwen3-8b", enable_thinking=True)

# Function to generate different responses using slight prompt variations
def generate_candidate_responses(sample: DataSample, n: int = 5) -> DataSample:
    """Generate multiple candidate responses for Best-of-N selection."""
    base_prompt = sample.input[0].content

    # Generate N variations of the prompt to get diverse responses
    for i in range(n):
        variation = f"{base_prompt} (Variation {i+1})" if i > 0 else base_prompt

        # Add some randomness to the prompt to encourage diversity
        if i == 1:
            variation += " Use bullet points."
        elif i == 2:
            variation += " Be very concise."
        elif i == 3:
            variation += " Include specific examples."
        elif i == 4:
            variation += " Use a conversational tone."

        # Generate response
        response = llm.simple_chat(variation)

        # Add to output
        sample.output.append(DataOutput(answer=Step(content=response)))

    return sample
```


```python
# Generate 5 candidate responses
sample = generate_candidate_responses(sample, n=5)

# Print generated responses
print("Generated Candidate Responses:")
for i, response in enumerate(sample.output):
    print(f"\n{i+1}. {response.answer.content[:200]}...")
```

## 5. Select the Best Response

Using the [best_of_n](../rm_gallery/core/reward/base.py#L139-L165) method from the reward model, we can select the top response(s).


```python
# Load a built-in reward model
reward = RewardRegistry.get("base_helpfulness_listwise")(
    name="helpfulness",
    llm=llm,
    rubrics=["Judge according to your own standard"]
)
# Get the best response
best_sample = reward.best_of_n(sample=sample, n=1)

print("\n🏆 Best Response:")
print(f"Score: {best_sample.output[0].answer.reward.score:.2f}")
print(f"\nContent:\n{best_sample.output[0].answer.content}")
```

## 6. Full Workflow Example

Let's put it all together into a reusable function.


```python
def best_of_n_pipeline(prompt: str, n_candidates: int = 5, n_best: int = 1) -> DataSample:
    """Full pipeline for Best-of-N response selection."""
    # Create initial sample
    sample = DataSample(
        unique_id="best_of_n_pipeline",
        input=[ChatMessage(role="user", content=prompt)],
        output=[]
    )

    # Generate candidate responses
    sample = generate_candidate_responses(sample, n=n_candidates)

    # Select best response
    best_sample = reward.best_of_n(sample, n=n_best)

    return best_sample
```


```python
# Try the full pipeline
best_response = best_of_n_pipeline("What are the benefits of regular exercise?", n_candidates=5, n_best=1)

print("\n🏆 Final Selected Response:")
print(best_response.output[0].answer.content)
```

## 7. Real-world Applications

The Best-of-N approach can be applied in various scenarios such as:

- Content moderation systems
- Customer service chatbots
- Educational assistants
- Code generation tools
- Creative writing assistance

For production environments, you might want to:
- Cache generated responses
- Implement rate limiting
- Add monitoring and logging
- Set up fallback mechanisms
- Optimize for latency and cost
