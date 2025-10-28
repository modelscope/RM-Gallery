# LLM Response Refinement Tutorial
## 1. Overview


This tutorial demonstrates how to use the [LLMRefinement](../../../rm_gallery/core/reward/refinement.py#L12-L131) class for iterative improvement of LLM responses using reward model feedback.

For more advanced usage, such as iterative refinement with comprehensive evaluation to correct datasamples, see [data_correction](../../../examples/rm_application/data_correction.py).

Key Concepts:

- **Iterative Refinement**: Repeatedly improve responses through feedback loops

- **Reward Model Feedback**: Use reward model assessments to guide improvements

- **Response Evolution**: Maintain response history to enable refinement

- **Dynamic Prompting**: Construct prompts based on feedback and history

## 2. Setup

First, let's import necessary modules:


```python
# Import core modules
import sys
sys.path.append("../../..")

from rm_gallery.core.data.schema import DataSample, DataOutput, Step, ChatMessage
from rm_gallery.core.model.message import MessageRole
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.reward.base import BaseLLMReward
from rm_gallery.core.reward.refinement import LLMRefinement
from loguru import logger
import uuid
```

## 3. Create Sample Input

Let's start by creating a sample input to work with.


```python
# Create a sample input
sample = DataSample(
    unique_id="refinement_demo",
    input=[
        ChatMessage(
            role=MessageRole.USER,
            content="Explain quantum computing in simple terms"
        )
    ],
    output=[]  # We'll generate responses later
)
```

## 4. Initialize Reward

We'll initialize our reward.


```python
# Initialize LLM for response generation
llm = OpenaiLLM(model="qwen3-8b", enable_thinking=True)

# Initialize reward model
reward: BaseLLMReward = RewardRegistry.get("base_helpfulness_listwise")(
    name="helpfulness",
    llm=llm
)
```

## 5. Run Refinement Process

We will give two examples of how to run the refinement process.

### 5.1. Run in Reward


```python
result = refined_sample = reward.refine(sample, max_iterations=3)
print("\n🏆 Final Refined Response:")
print(result)
```

### 5.2. Run in Refinement


```python
# Create refinement module
refiner = LLMRefinement(
    llm=llm,
    reward=reward,
    max_iterations=3
)

result = refiner.run(sample)
print("\n🏆 Final Refined Response:")
print(result)
```

## 6. Detailed Analysis

Let's look at what happens during each iteration of the refinement process.


```python
def detailed_run(sample: DataSample, max_iterations: int = 3):
    """Run refinement process with detailed output for each iteration."""

    # Initial response generation
    response = llm.chat(sample.input)
    sample.output.append(DataOutput(answer=Step(
        role=MessageRole.ASSISTANT,
        content=response.content
    )))

    print("Initial Response:")
    print(response.content)
    print("\n" + "-" * 50 + "\n")

    # Iterative refinement loop
    for i in range(max_iterations):

        # Generate feedback
        feedback = refiner._generate_feedback(sample)

        # Print iteration details
        print(f"Iteration {i+1}/{max_iterations}:")
        print("Feedback Received:", feedback)

        # Generate refined response
        sample = refiner._generate_response(sample, feedback)

        print("Refined Response:")
        print(sample.output[-1].answer.content)
        print("\n" + "-" * 50 + "\n")

    return sample.output[-1].answer.content
```


```python
# Run with detailed analysis

sample = DataSample(
    unique_id="detailed_run_demo",
    input=[
        ChatMessage(
            role=MessageRole.USER,
            content="What are the benefits of regular exercise?"
        )
    ],
    output=[]  # We'll generate responses later
)
detailed_run(sample)
```

## 7. Real-world Applications

The refinement approach can be applied in various scenarios such as:

- Academic writing assistance
- Technical documentation improvement
- Educational content creation
- Code explanation refinement
- Research summarization
- Business communication optimization

For production environments, you might want to:
- Implement caching for intermediate responses
- Add comprehensive error handling
- Set up detailed logging
- Implement batch processing capabilities
