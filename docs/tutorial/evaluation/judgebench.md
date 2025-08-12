# JudgeBench Evaluation Tutorial

## Overview

JudgeBench is a comprehensive benchmark for evaluating LLM-based judges. It tests the ability of reward models to perform pairwise comparisons and rank responses effectively. This tutorial demonstrates how to use RM-Gallery's JudgeBench evaluator to assess your reward models' judging capabilities.

The JudgeBench evaluation protocol supports multiple judge types including Vanilla, Arena-Hard, AutoJ, Prometheus2, and Skywork-Critic, each with their own specialized evaluation templates and parsing logic.

## Features

- **Multiple Judge Types**: Supports 5 different judge types with specialized evaluation protocols
- **Pairwise Comparison**: Evaluates model ability to compare two responses and determine which is better
- **Template System**: Uses Jinja2 templates for flexible prompt generation
- **Position Bias Mitigation**: Optional response shuffling to reduce position-based biases
- **Source-wise Analysis**: Provides accuracy metrics broken down by data source
- **Parallel Processing**: Supports multi-threaded evaluation for faster processing

## Data Preparation

### Step 1: Download the Dataset

First, create the data directory and clone the JudgeBench repository:

```bash
# Create the benchmark data directory
mkdir -p data/benchmarks

# Navigate to the directory
cd data/benchmarks

# Clone the JudgeBench repository from GitHub
git clone https://github.com/ScalerLab/JudgeBench.git
```

### Step 2: Verify Data Structure

After cloning, your data structure should look like:

```
data/
└── benchmarks/
    └── JudgeBench/
        ├── data/
        │   ├── dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl
        │   ├── dataset=judgebench,response_model=claude-3.5-sonnet-20240620.jsonl
        │   └── ...
        ├── utils/
        │   ├── templates/
        │   │   ├── arena_hard_judge_prompt.jinja2
        │   │   ├── vanilla_prompt.jinja2
        │   │   ├── autoj_prompt.jinja2
        │   │   ├── prometheus2_prompt.jinja2
        │   │   ├── skywork_critic_prompt.jinja2
        │   │   └── ...
        │   └── ...
        └── README.md
```

### Step 3: Template Files

The JudgeBench evaluation relies on Jinja2 template files located in `data/benchmarks/JudgeBench/utils/templates/`. These templates define the evaluation prompts for different judge types.

## Environment Setup

### Prerequisites

Ensure you have the required environment variables set up for your language model:

```bash
# For OpenAI-compatible APIs
export OPENAI_API_KEY="your_api_key_here"
export BASE_URL="your_base_url_here"  # Optional, for custom endpoints
```

**Environment Variables Check:**
```bash
# Verify environment variables are set
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:8}..."  # Shows first 8 characters
echo "BASE_URL: $BASE_URL"

# Or check if they exist
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY is not set"
else
    echo "✅ OPENAI_API_KEY is set"
fi
```

### Installation

Make sure RM-Gallery is installed with the required dependencies:

```bash
pip install rm-gallery
pip install jinja2  # Required for template rendering
```

### Quick Installation Check

Verify your installation by running the help command:

```bash
python rm_gallery/gallery/evaluation/judgebench.py --help
```

This should display the available command-line options. If you see an error about missing modules, ensure all dependencies are installed correctly.

## Basic Usage

### Quick Start

The easiest way to run JudgeBench evaluation is directly from the command line:

```bash
# Simplest command with default parameters
python rm_gallery/gallery/evaluation/judgebench.py

# Or with custom parameters
python rm_gallery/gallery/evaluation/judgebench.py \
    --data_path "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
    --result_path "data/results/judgebench_results.json" \
    --judge_type "arena_hard" \
    --max_samples 50 \
    --model "deepseek-chat" \
    --max_workers 4
```

**Expected Output:**
```
Evaluation completed!
Judge type: arena_hard
Model: deepseek-chat
Overall accuracy: 68.00%
Valid samples: 50
Total samples: 50

Accuracy by source:
  mmlu-pro-computer science: 75.00%
  gsm8k: 60.00%
```

### Command Line Parameters

All parameters are optional and have default values:

```bash
# Minimal command with defaults
python rm_gallery/gallery/evaluation/judgebench.py

# Full command with all parameters
python rm_gallery/gallery/evaluation/judgebench.py \
    --data_path "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
    --result_path "data/results/judgebench.json" \
    --judge_type "arena_hard" \
    --max_samples 10 \
    --model "deepseek-chat" \
    --max_workers 4 \
    --shuffle_responses
```

### Programmatic Usage

You can also use the evaluation in your Python code:

```python
from rm_gallery.gallery.evaluation.judgebench import main

# Run evaluation with custom settings
main(
    data_path="data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
    result_path="data/results/judgebench_results.json",
    judge_type="arena_hard",
    max_samples=50,
    model="deepseek-chat",
    max_workers=4
)
```

### Important Notes

1. **Data Path**: Make sure the data path points to a valid JudgeBench dataset file. The default path assumes you've downloaded the dataset to `data/benchmarks/JudgeBench/`.

2. **Results Directory**: The script will create the results directory if it doesn't exist. Make sure you have write permissions.

3. **Model Configuration**: For simple model names, use the `--model` parameter. For complex configurations (temperature, max_tokens, etc.), use the programmatic approach.

4. **Environment Variables**: Ensure `OPENAI_API_KEY` and optionally `BASE_URL` are set before running the evaluation.

## Judge Types

JudgeBench supports multiple judge types, each with specialized evaluation protocols:

### 1. Vanilla Judge
- **Type**: `vanilla`
- **Description**: Simple binary choice evaluation
- **Output Format**: "Output (a)" or "Output (b)"
- **Use Case**: Basic preference evaluation

### 2. Arena-Hard Judge
- **Type**: `arena_hard`
- **Description**: Advanced comparison with system/user message format
- **Output Format**: `[[A>B]]`, `[[B>A]]`, or `[[A=B]]`
- **Use Case**: Comprehensive response evaluation

### 3. AutoJ Judge
- **Type**: `auto_j`
- **Description**: Automated judgment with detailed reasoning
- **Output Format**: "final decision is Response 1/2/tie"
- **Use Case**: Detailed comparative analysis

### 4. Prometheus2 Judge
- **Type**: `prometheus_2`
- **Description**: Rubric-based evaluation framework
- **Output Format**: `[RESULT] A` or `[RESULT] B`
- **Use Case**: Criteria-based assessment

### 5. Skywork-Critic Judge
- **Type**: `skywork_critic`
- **Description**: Specialized critic evaluation
- **Output Format**: `[[A]]` or `[[B]]`
- **Use Case**: Critical analysis of responses

## Advanced Usage

### Custom Model Configuration

You can pass custom model configurations for different judge types:

```python
from rm_gallery.gallery.evaluation.judgebench import main

# Custom model configuration
model_config = {
    "model": "qwen3-32b",
    "temperature": 0.1,
    "max_tokens": 2048,
    "enable_thinking": True
}

main(
    data_path="data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
    result_path="data/results/judgebench_custom.json",
    judge_type="prometheus_2",
    max_samples=100,
    model=model_config,
    max_workers=8
)
```

### Programmatic Usage

For more control over the evaluation process:

```python
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.gallery.evaluation.judgebench import JudgeBenchEvaluator, JudgeBenchReward
from rm_gallery.core.utils.file import write_json

# 1. Set up data loading
config = {
    "path": "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
    "limit": 50,
}

load_module = create_loader(
    name="judgebench",
    load_strategy_type="local",
    data_source="judgebench",
    config=config,
)

# 2. Initialize language model
llm = OpenaiLLM(model="deepseek-chat", enable_thinking=True)

# 3. Load dataset
dataset = load_module.run()

# 4. Create evaluator with specific judge type
evaluator = JudgeBenchEvaluator(
    reward=JudgeBenchReward(
        name="judgebench_eval",
        judge_type="arena_hard",  # Choose your judge type
        llm=llm,
        max_workers=4,
    )
)

# 5. Run evaluation with optional response shuffling
results = evaluator.run(
    samples=dataset.get_data_samples(),
    shuffle_responses=True  # Enable to reduce position bias
)

# 6. Save results
write_json(results, "data/results/judgebench_detailed.json")

# 7. Print summary
print(f"Judge Type: {results['judge_type']}")
print(f"Overall Accuracy: {results['overall_accuracy']['accuracy']:.4f}")
print(f"Valid Samples: {results['overall_accuracy']['valid_samples']}")
```

### Using Custom Templates

For research purposes or custom evaluation scenarios, you can define and use custom templates:

```python
from rm_gallery.gallery.evaluation.judgebench import JudgeBenchBaseTemplate, JudgeBenchReward, JudgeBenchEvaluator
from rm_gallery.core.model.openai_llm import OpenaiLLM
from pydantic import Field

class ResearchJudgeTemplate(JudgeBenchBaseTemplate):
    """Custom template for research purposes"""
    
    decision: str = Field(default="", description="Research decision format")
    confidence: float = Field(default=0.0, description="Confidence score")
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        # Use direct prompt instead of template file
        return f"""
As a research evaluator, compare these two responses:

Question: {question}

Response A: {answer_A}

Response B: {answer_B}

Instructions:
1. Evaluate based on accuracy, completeness, and clarity
2. Provide your decision as [[A>B]], [[B>A]], or [[A=B]]
3. Include confidence level (0.0-1.0)

Format your response as:
Decision: [[your_decision]]
Confidence: your_confidence_score
"""
    
    @classmethod
    def parse(cls, text: str) -> "ResearchJudgeTemplate":
        decision = ""
        confidence = 0.0
        
        # Parse decision - MUST return A>B, B>A, A=B, or empty string
        if "[[A>B]]" in text:
            decision = "A>B"  # ✅ Valid format
        elif "[[B>A]]" in text:
            decision = "B>A"  # ✅ Valid format
        elif "[[A=B]]" in text:
            decision = "A=B"  # ✅ Valid format
        else:
            decision = ""  # ✅ Valid - unable to parse
        
        # Parse confidence
        import re
        conf_match = re.search(r"Confidence:\s*([0-9.]+)", text)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.0
        
        return cls(decision=decision, confidence=confidence)

# Use the custom template
llm = OpenaiLLM(model="deepseek-chat")
evaluator = JudgeBenchEvaluator(
    reward=JudgeBenchReward(
        name="research_judge",
        judge_type="research",  # Custom type
        template=ResearchJudgeTemplate,  # Use custom template
        llm=llm,
        max_workers=4,
    )
)

# Run evaluation
results = evaluator.run(samples=dataset.get_data_samples())
```

## Configuration Parameters

### Main Function Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | Required | Path to the JudgeBench dataset file |
| `result_path` | str | Required | Path to save evaluation results |
| `judge_type` | str | "arena_hard" | Type of judge to use |
| `max_samples` | int | 10 | Maximum number of samples to evaluate |
| `model` | str/dict | "deepseek-chat" | Model name or configuration dictionary |
| `max_workers` | int | 4 | Number of parallel workers for evaluation |
| `shuffle_responses` | bool | False | Whether to shuffle response order |

### JudgeBenchReward Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Name of the reward instance |
| `judge_type` | str | "arena_hard" | Type of judge to use |
| `template` | Type[JudgeBenchBaseTemplate] | None | Custom template class (overrides judge_type mapping) |
| `llm` | OpenaiLLM | Required | Language model instance |
| `max_workers` | int | 4 | Number of parallel workers for evaluation |

### Judge Type Options

| Judge Type | Description | Output Format |
|------------|-------------|---------------|
| `vanilla` | Simple binary choice | "Output (a)" or "Output (b)" |
| `arena_hard` | Advanced comparison | `[[A>B]]`, `[[B>A]]`, `[[A=B]]` |
| `auto_j` | Automated judgment | "final decision is Response 1/2/tie" |
| `prometheus_2` | Rubric-based evaluation | `[RESULT] A` or `[RESULT] B` |
| `skywork_critic` | Specialized critic | `[[A]]` or `[[B]]` |

## Understanding Results

### Result Structure

The evaluation results contain:

```json
{
    "model": "deepseek-chat",
    "judge_type": "arena_hard",
    "overall_accuracy": {
        "accuracy": 0.68,
        "correct_count": 34,
        "valid_samples": 50,
        "total_samples": 50
    },
    "source_accuracy": {
        "mmlu-pro-computer science": {
            "accuracy": 0.75,
            "correct_count": 15,
            "valid_samples": 20,
            "total_samples": 20
        },
        "gsm8k": {
            "accuracy": 0.60,
            "correct_count": 18,
            "valid_samples": 30,
            "total_samples": 30
        }
    }
}
```

### Key Metrics

- **accuracy**: Proportion of correct judgments (0.0 to 1.0)
- **correct_count**: Number of correct pairwise comparisons
- **valid_samples**: Number of successfully processed samples
- **source_accuracy**: Accuracy broken down by data source

### Data Sample Format

Each sample in JudgeBench contains:

```json
{
    "pair_id": "unique-identifier",
    "original_id": "source-question-id",
    "source": "mmlu-pro-computer science",
    "question": "Question content...",
    "response_model": "gpt-4o-2024-05-13",
    "response_A": "First response...",
    "response_B": "Second response...",
    "label": "A>B"  // Ground truth label
}
```

## Troubleshooting

### Common Issues

1. **Missing Template Files**:
   ```bash
   # Ensure JudgeBench templates are in the correct location
   ls data/benchmarks/JudgeBench/utils/templates/
   ```

2. **Template Rendering Errors**:
   ```bash
   pip install jinja2
   ```

3. **Judge Type Not Recognized**:
   - Verify the judge type is one of: `vanilla`, `arena_hard`, `auto_j`, `prometheus_2`, `skywork_critic`

4. **Data Format Issues**:
   - Ensure the dataset follows the JudgeBench format
   - Check that each sample has exactly 2 responses for comparison

5. **Custom Template Issues**:
   - Verify your template's `parse` method returns `decision` in the correct format: `"A>B"`, `"B>A"`, `"A=B"`, or `""`
   - Check that your template inherits from `JudgeBenchBaseTemplate`
   - Ensure all required fields are defined with proper Pydantic Field annotations
   - Test your template's parsing logic with sample model outputs

### Performance Optimization

1. **Parallel Processing**: Increase `max_workers` for faster evaluation
2. **Sample Limiting**: Use `max_samples` to test with smaller datasets first
3. **Response Shuffling**: Enable `shuffle_responses=True` to reduce position bias

## Best Practices

1. **Judge Selection**: Choose appropriate judge types for your evaluation needs
2. **Position Bias**: Always test with `shuffle_responses=True` for fair evaluation
3. **Sample Size**: Start with small samples (10-50) for testing, then scale up
4. **Source Analysis**: Analyze performance across different data sources
5. **Multiple Runs**: Run evaluation multiple times with different random seeds
6. **Custom Template Testing**: Always test your custom templates with sample outputs before full evaluation
7. **Template Validation**: Implement robust parsing logic that handles edge cases and unexpected model outputs
8. **Error Handling**: Return empty decision string (`""`) when parsing fails, rather than raising exceptions

## Examples

### Example 1: Compare Multiple Judge Types

```bash
# Run evaluations for different judge types
judge_types=("vanilla" "arena_hard" "auto_j" "prometheus_2" "skywork_critic")

for judge_type in "${judge_types[@]}"; do
    echo "Running $judge_type evaluation..."
    python rm_gallery/gallery/evaluation/judgebench.py \
        --data_path "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
        --result_path "data/results/judgebench_${judge_type}.json" \
        --judge_type "$judge_type" \
        --max_samples 20 \
        --model "deepseek-chat" \
        --max_workers 4
done
```

Or using Python to loop through judge types:

```python
import subprocess
import sys

judge_types = ["vanilla", "arena_hard", "auto_j", "prometheus_2", "skywork_critic"]

for judge_type in judge_types:
    print(f"Running {judge_type} evaluation...")
    
    cmd = [
        sys.executable, "rm_gallery/gallery/evaluation/judgebench.py",
        "--data_path", "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
        "--result_path", f"data/results/judgebench_{judge_type}.json",
        "--judge_type", judge_type,
        "--max_samples", "20",
        "--model", "deepseek-chat",
        "--max_workers", "4"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {judge_type} evaluation completed successfully")
    else:
        print(f"❌ {judge_type} evaluation failed: {result.stderr}")
```

### Example 2: Production Evaluation with Position Bias Mitigation

```bash
# Full evaluation with bias mitigation
python rm_gallery/gallery/evaluation/judgebench.py \
    --data_path "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
    --result_path "data/results/judgebench_production.json" \
    --judge_type "arena_hard" \
    --max_samples 200 \
    --model "qwen3-32b" \
    --max_workers 8 \
    --shuffle_responses
```

For more complex model configurations, you can still use the programmatic approach:

```python
from rm_gallery.gallery.evaluation.judgebench import main

# Full evaluation with custom model configuration
main(
    data_path="data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
    result_path="data/results/judgebench_production.json",
    judge_type="arena_hard",
    max_samples=200,
    model={
        "model": "qwen3-32b",
        "temperature": 0.1,
        "enable_thinking": True
    },
    max_workers=8,
    shuffle_responses=True
)
```

### Example 3: Testing Custom Templates

```python
from rm_gallery.gallery.evaluation.judgebench import JudgeBenchBaseTemplate, JudgeBenchReward
from rm_gallery.core.model.openai_llm import OpenaiLLM
from pydantic import Field

# Define and test a custom template
class TestCustomTemplate(JudgeBenchBaseTemplate):
    decision: str = Field(default="", description="Decision in A>B format")
    confidence: float = Field(default=0.0, description="Confidence score")
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        return f"""
Evaluate these responses:
Q: {question}
A: {answer_A}
B: {answer_B}

Which is better? Format: Decision: [[A>B/B>A/A=B]] Confidence: 0.X
"""
    
    @classmethod
    def parse(cls, text: str) -> "TestCustomTemplate":
        decision = ""
        confidence = 0.0
        
        # Test different parsing patterns
        test_patterns = [
            (r"\[\[A>B\]\]", "A>B"),
            (r"\[\[B>A\]\]", "B>A"),
            (r"\[\[A=B\]\]", "A=B"),
            (r"Decision:\s*A>B", "A>B"),
            (r"Decision:\s*B>A", "B>A"),
            (r"Decision:\s*A=B", "A=B"),
        ]
        
        import re
        for pattern, result in test_patterns:
            if re.search(pattern, text):
                decision = result
                break
        
        # Parse confidence
        conf_match = re.search(r"Confidence:\s*([0-9.]+)", text)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.0
        
        return cls(decision=decision, confidence=confidence)

# Test the template parsing
def test_template_parsing():
    template = TestCustomTemplate
    
    test_cases = [
        "Decision: [[A>B]] Confidence: 0.8",
        "I think [[B>A]] is better. Confidence: 0.9",
        "Both are equal [[A=B]] Confidence: 0.5",
        "Unclear response without clear format",
    ]
    
    print("Testing custom template parsing:")
    for i, test_input in enumerate(test_cases):
        result = template.parse(test_input)
        print(f"Test {i+1}: '{test_input[:30]}...'")
        print(f"  Decision: '{result.decision}' (Valid: {result.decision in ['A>B', 'B>A', 'A=B', '']})")
        print(f"  Confidence: {result.confidence}")
        print()

# Run the test
test_template_parsing()

# Use the tested template in evaluation
llm = OpenaiLLM(model="deepseek-chat")
reward = JudgeBenchReward(
    name="tested_custom_judge",
    template=TestCustomTemplate,
    llm=llm
)
print("Custom template ready for evaluation!")
```

### Example 4: Detailed Analysis with Custom Processing

```python
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.gallery.evaluation.judgebench import JudgeBenchEvaluator, JudgeBenchReward
import json

# Load and process results
def analyze_judge_performance(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Judge Type: {results['judge_type']}")
    print(f"Model: {results['model']}")
    print(f"Overall Accuracy: {results['overall_accuracy']['accuracy']:.2%}")
    
    # Analyze by source
    print("\nPerformance by Source:")
    for source, metrics in results['source_accuracy'].items():
        print(f"  {source}: {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['valid_samples']})")

# Run analysis
analyze_judge_performance("data/results/judgebench_production.json")
```

## Template Customization

> **⚠️ Important**: Before creating custom templates, please read the [Template Constraints](#️-important-constraints-for-custom-templates) section below. Your custom template's `parse` method must return decision values in the exact format expected by the evaluation system.

### Using Custom Templates

You can now pass custom template classes directly to the `JudgeBenchReward` constructor:

```python
from rm_gallery.gallery.evaluation.judgebench import JudgeBenchReward, JudgeBenchBaseTemplate
from rm_gallery.core.model.openai_llm import OpenaiLLM
from pydantic import Field

class CustomJudgeTemplate(JudgeBenchBaseTemplate):
    decision: str = Field(default="", description="Custom decision format")
    
    @classmethod
    def format(cls, question: str, answer_A: str, answer_B: str, **kwargs) -> str:
        renderer = cls.get_renderer()
        return renderer.render_template(
            "custom_judge_prompt",  # Create this template file
            question=question,
            answer_a=answer_A,
            answer_b=answer_B
        )
    
    @classmethod
    def parse(cls, text: str) -> "CustomJudgeTemplate":
        # Implement custom parsing logic
        decision = ""  # Parse from text
        return cls(decision=decision)

# Use custom template
llm = OpenaiLLM(model="deepseek-chat")
reward = JudgeBenchReward(
    name="custom_judge",
    judge_type="custom",  # Can be any string when using custom template
    template=CustomJudgeTemplate,  # Pass your custom template class
    llm=llm
)
```

### ⚠️ Important Constraints for Custom Templates

**Critical**: Custom templates must conform to the existing evaluation processing logic. Your template's `parse` method must return an object with the following structure:

#### Required Fields

- **`decision`** (str): Must be one of the following values:
  - `"A>B"` - Response A is better than Response B
  - `"B>A"` - Response B is better than Response A  
  - `"A=B"` - Both responses are equal
  - `""` (empty string) - Unable to make a decision

#### Optional Fields

- **`reason`** (str): Human-readable explanation of the decision. If not provided, a default reason will be generated.

#### How the Evaluation System Processes Decisions

The evaluation system converts your template's decision output into numerical scores as follows:

```python
# Internal processing logic in _after_evaluate method
if decision == "A>B":
    scores = [1, 0]  # Response A gets score 1, Response B gets score 0
elif decision == "B>A":
    scores = [0, 1]  # Response A gets score 0, Response B gets score 1
elif decision == "A=B":
    scores = [0.5, 0.5]  # Both responses get equal scores
else:  # Empty string or invalid format
    scores = [0, 0]  # Both responses get zero scores (evaluation failed)
```

These scores are then used to compute accuracy by comparing against the ground truth labels in the dataset.

#### Example of Correct Implementation

```python
class ValidCustomTemplate(JudgeBenchBaseTemplate):
    decision: str = Field(default="", description="Must be A>B, B>A, A=B, or empty")
    reason: str = Field(default="", description="Optional explanation")
    
    @classmethod
    def parse(cls, text: str) -> "ValidCustomTemplate":
        decision = ""
        reason = ""
        
        # Your parsing logic MUST produce one of these decision values
        if "Response A is better" in text:
            decision = "A>B"
            reason = "Response A provided more accurate information"
        elif "Response B is better" in text:
            decision = "B>A"
            reason = "Response B was more comprehensive"
        elif "Both responses are equal" in text:
            decision = "A=B"
            reason = "Both responses have similar quality"
        else:
            decision = ""  # Unable to parse
            reason = "Could not determine preference"
        
        return cls(decision=decision, reason=reason)
```

#### Common Mistakes to Avoid

```python
# ❌ WRONG: Invalid decision format
class InvalidTemplate(JudgeBenchBaseTemplate):
    decision: str = Field(default="", description="Decision")
    
    @classmethod
    def parse(cls, text: str) -> "InvalidTemplate":
        # These values will NOT work with the evaluation system
        if "first" in text:
            decision = "first"  # ❌ Invalid - should be "A>B"
        elif "second" in text:
            decision = "second"  # ❌ Invalid - should be "B>A"
        elif "winner: A" in text:
            decision = "A"  # ❌ Invalid - should be "A>B"
        
        return cls(decision=decision)

# ✅ CORRECT: Valid decision format
class ValidTemplate(JudgeBenchBaseTemplate):
    decision: str = Field(default="", description="Decision")
    
    @classmethod
    def parse(cls, text: str) -> "ValidTemplate":
        decision = ""
        if "first" in text:
            decision = "A>B"  # ✅ Valid format
        elif "second" in text:
            decision = "B>A"  # ✅ Valid format
        elif "winner: A" in text:
            decision = "A>B"  # ✅ Valid format
        
        return cls(decision=decision)
```

### Backward Compatibility

The existing `judge_type` parameter still works as before:

```python
# Using predefined judge types (backward compatible)
reward = JudgeBenchReward(
    name="arena_hard_judge",
    judge_type="arena_hard",  # Uses ArenaHardTemplate automatically
    llm=llm
)

# Custom template overrides judge_type mapping
reward = JudgeBenchReward(
    name="custom_judge",
    judge_type="arena_hard",  # This will be ignored
    template=CustomJudgeTemplate,  # This takes precedence
    llm=llm
)
```

### Creating Template Files

If your custom template references a Jinja2 template file, create it in the templates directory:

```bash
# Create your custom template file
touch data/benchmarks/JudgeBench/utils/templates/custom_judge_prompt.jinja2
```

Example template content:

```jinja2
Please evaluate the following two responses to determine which is better.

Question: {{ question }}

Response A: {{ answer_a }}

Response B: {{ answer_b }}

Which response is better? Reply with [[A]] or [[B]].
```

## Integration with Other Benchmarks

JudgeBench can be integrated with other evaluation frameworks:

```bash
# Example: Combining with RewardBench2
echo "Running RewardBench2 evaluation..."
python rm_gallery/gallery/evaluation/rewardbench2.py \
    --data_path "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet" \
    --result_path "data/results/rewardbench2.json" \
    --max_samples 100 \
    --model "deepseek-chat"

echo "Running JudgeBench evaluation..."
python rm_gallery/gallery/evaluation/judgebench.py \
    --data_path "data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl" \
    --result_path "data/results/judgebench.json" \
    --judge_type "arena_hard" \
    --max_samples 100 \
    --model "deepseek-chat"

echo "Both evaluations completed!"
```

Or using Python to run both evaluations:

```python
import subprocess
import sys

# Example: Combining with RewardBench2
def run_evaluation(script_path, **kwargs):
    """Helper function to run evaluation scripts"""
    cmd = [sys.executable, script_path]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {script_path} completed successfully")
    else:
        print(f"❌ {script_path} failed: {result.stderr}")
    return result.returncode == 0

# Run both evaluations
print("Running RewardBench2 evaluation...")
rewardbench2_success = run_evaluation(
    "rm_gallery/gallery/evaluation/rewardbench2.py",
    data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path="data/results/rewardbench2.json",
    max_samples=100,
    model="deepseek-chat"
)

print("Running JudgeBench evaluation...")
judgebench_success = run_evaluation(
    "rm_gallery/gallery/evaluation/judgebench.py",
    data_path="data/benchmarks/JudgeBench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
    result_path="data/results/judgebench.json",
    judge_type="arena_hard",
    max_samples=100,
    model="deepseek-chat"
)

if rewardbench2_success and judgebench_success:
    print("🎉 Both evaluations completed successfully!")
else:
    print("⚠️  Some evaluations failed. Check the logs above.")
```

## Next Steps

After running JudgeBench evaluation:

1. **Analyze Results**: Compare performance across different judge types
2. **Source Analysis**: Identify strengths and weaknesses by data source
3. **Bias Testing**: Run with and without response shuffling to detect bias
4. **Model Comparison**: Evaluate different models using the same judge type
5. **Integration**: Integrate evaluation into your model development pipeline

For more evaluation benchmarks and advanced scenarios, check out other evaluation tutorials in the RM-Gallery documentation. 