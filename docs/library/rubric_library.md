# Rubric Library

!!! info "Coming Soon"
    This page is under construction. We are working on documenting all available evaluation rubrics.

## Overview

The Rubric Library provides a collection of evaluation principles and rubrics that can be used to guide reward model assessments.

## What is a Rubric?

A rubric is a set of evaluation principles or criteria used to assess the quality of model outputs. In RM-Gallery, rubrics follow the **Principle-Critic-Score** paradigm.

## Available Rubrics

### General Purpose Rubrics
- Coming soon...

### Task-Specific Rubrics
- Coming soon...

### Domain-Specific Rubrics
- Coming soon...

## Creating Custom Rubrics

For information on creating custom rubrics and principles, see:
- [Auto Rubric Generation](../tutorial/building_rm/autoprinciple.ipynb)
- [Custom Reward Models](../tutorial/building_rm/custom_reward.ipynb)

## Usage Example

```python
from rm_gallery.core.reward import BaseListWisePrincipleReward
from rm_gallery.core.model.openai_llm import OpenaiLLM
import os

# Set up your LLM
os.environ["OPENAI_API_KEY"] = "your_api_key"
os.environ["BASE_URL"] = "your_base_url"

llm = OpenaiLLM(model="qwen3-8b", enable_thinking=True)

# Create a reward model with custom principles
reward = BaseListWisePrincipleReward(
    name="custom_rubric_reward",
    desc="Your task description",
    scenario="Your scenario description",
    principles=[
        "Principle 1: Clear and accurate responses",
        "Principle 2: Helpful and relevant information",
        "Principle 3: Safe and ethical content"
    ],
    llm=llm
)
```


