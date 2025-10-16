# Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling

## 🚀 Key Features

- **🎯 Training-Free**: No parameter updates required - works with any pre-trained LLM
- **📊 Data Efficient**: Achieves SOTA performance using only ~70 preference pairs (1.5% of source data)
- **🔍 Interpretable**: Generates human-readable "Theme-Tips" rubric hierarchies
- **⚡ Fast Convergence**: Information-theoretic selection rapidly identifies optimal rubric sets
- **🌐 Cross-Model**: Rubrics generalize across different LLM architectures
- **🔄 Modular Pipeline**: Separate generation, structuring, and analysis components

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Configuration Guide](#configuration-guide)
- [Data Format](#data-format)
- [Advanced Usage](#advanced-usage)

## 🚀 Quick Start

Navigate to the examples directory:
```bash
cd examples/rubric/
```

### Option 1: Complete Auto-Rubric Pipeline

```bash
# 🎯 Run the complete Auto-Rubric pipeline (Generation + MCR² + Structuring)
./run_autorubric.sh
```

This will:
1. Generate rubrics from preference data
2. Apply MCR² selection for optimal rubric sets
3. Structure rubrics into Theme-Tips format
4. Export results to `./exports/{model_name}/`

### Option 2: Step-by-Step Pipeline

```bash
# Step 1: Generate Rubrics
./run_generator.sh

# Step 2: Structure into Theme-Tips
./run_structurer.sh

# Step 3: Analyze Performance
./run_analysis.sh
```

### Quick Configuration

Edit the shell scripts to customize parameters:

**`run_autorubric.sh`** - Complete pipeline:
```bash
MODEL="qwen3-32b"
MAX_SAMPLES=200        # Adjust based on your data size
MAX_WORKERS=32         # Adjust based on your hardware
NUM_CATEGORIES=5       # Number of Theme-Tips categories
```

**`run_generator.sh`** - Rubric generation:
```bash
MAX_SAMPLES=200        # Number of samples to process
DOMAINS="multilingual" # Filter by domain (or remove for all)
BATCH_SIZE=500         # Batch size for processing
```

## 🏗️ Pipeline Components

### 1. Complete Auto-Rubric Pipeline (`auto_rubric.py`)

The integrated pipeline combining generation, MCR² selection, and structuring:

```bash
# Run complete pipeline
python auto_rubric.py \
    --data-path data/helpsteer3_preference_train.jsonl \
    --model qwen3-32b \
    --max-workers 32 \
    --enable-structuring True \
    --num-categories 5
```

**Pipeline Stages:**
1. **Iterative Generation**: Propose-Evaluate-Revise loop for rubric creation
2. **MCR² Selection**: Information-theoretic filtering for optimal rubric diversity
3. **Theme-Tips Structuring**: Hierarchical organization into interpretable categories
4. **Export**: Structured results ready for evaluation

### 2. Rubric Generation (`run_rubric_generator.py`)

Standalone rubric generation with checkpoint support:

```bash
# Generate rubrics with checkpointing
python run_rubric_generator.py \
    --data-path data/helpsteer3_preference_train.jsonl \
    --output-dir rubric_generation_output \
    --model qwen3-32b \
    --max-samples 200 \
    --batch-size 500 \
    --resume  # Resume from checkpoint if interrupted
```

**Key Features:**
- **Checkpoint Support**: Resume interrupted generation
- **Batch Processing**: Efficient parallel processing
- **Domain Filtering**: Focus on specific content domains
- **Iterative Refinement**: Multi-epoch improvement cycles

### 3. Rubric Structuring (`run_rubric_structurer.py`)

Transform raw rubrics into Theme-Tips format:

```bash
# Structure rubrics into themes
python run_rubric_structurer.py \
    --input rubric_generation_output/rubrics.json \
    --output rubric_structuring_results \
    --themes 5 \
    --model qwen3-32b
```

**Output Format (Theme-Tips):**
```
Theme: Evaluate response accuracy and factual correctness
- Tip 1: Check for factual errors or misconceptions
- Tip 2: Verify claims against reliable sources
- Tip 3: Assess logical consistency of arguments
```

### 4. Performance Analysis (`run_rubric_analysis.py`)

Comprehensive evaluation of rubric performance:

```bash
# Analyze rubric performance
python run_rubric_analysis.py \
    --rubrics rubric_structuring_results/ready_to_use_rubrics.json \
    --dataset data/helpsteer3_preference_valid.jsonl \
    --max-samples 100 \
    --max-workers 256 \
    --output rubric_analysis_results
```

**Generated Metrics:**
- **Coverage**: Percentage of samples where rubrics provide clear preference
- **Precision**: Accuracy of rubric predictions vs. ground truth
- **Contribution**: Individual rubric impact on ensemble performance
- **Ensemble Accuracy**: Overall performance of rubric set

## ⚙️ Configuration Guide

### Complete Pipeline (`auto_rubric.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `"qwen3-32b"` | LLM model for all operations |
| `--max-workers` | `32` | Concurrent threads for parallel processing |
| `--batch-size` | `10` | Samples processed per batch |
| `--max-epochs` | `10` | Maximum refinement iterations per sample |
| `--mcr-batch-size` | `10` | MCR² selection batch size |
| `--min-increment-threshold` | `0.002` | Information gain stopping threshold |
| `--patience` | `2` | Consecutive low increments before stopping |
| `--max-iterations` | `50` | Maximum pipeline iterations |
| `--max-total-rubrics` | `200` | Final rubric set size limit |
| `--enable-structuring` | `True` | Enable Theme-Tips structuring |
| `--num-categories` | `5` | Number of Theme-Tips categories |

### Rubric Generation (`run_rubric_generator.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-path` | Required | Path to preference dataset (JSONL) |
| `--model` | `"qwen3-32b"` | LLM model for generation |
| `--max-samples` | `200` | Maximum samples to process (-1 for all) |
| `--domains` | `"multilingual"` | Filter by domain (or remove for all) |
| `--batch-size` | `500` | Batch size for processing |
| `--max-epochs` | `10` | Maximum refinement epochs |
| `--max-workers` | `256` | Worker threads |
| `--resume` | Flag | Resume from checkpoint |
| `--disable-checkpoint` | Flag | Disable checkpoint saving |

### Rubric Structuring (`run_rubric_structurer.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | Required | Input rubrics JSON file |
| `--output` | `"rubric_structuring_results"` | Output directory |
| `--model` | `"qwen3-32b"` | LLM model for structuring |
| `--themes` | `5` | Number of themes to generate |

### Performance Analysis (`run_rubric_analysis.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rubrics` | Required | Path to rubrics JSON file |
| `--dataset` | `helpsteer3_preference_valid.jsonl` | Validation dataset |
| `--model` | `"qwen3-32b"` | Model for evaluation |
| `--max-samples` | `100` | Maximum samples for evaluation |
| `--max-workers` | `256` | Worker threads for parallel processing |
| `--source-rubrics` | Optional | Source rubrics for comparison |

## 📊 Data Format & Processing

### Expected Input Format

Input preference data should be in JSONL format with the following structure:

```json
{
  "input": [{"role": "user", "content": "Your question here"}],
  "output": [
    {
      "answer": {
        "content": "Response A",
        "label": {"preference": "chosen", "is_preferred": true}
      }
    },
    {
      "answer": {
        "content": "Response B",
        "label": {"preference": "rejected", "is_preferred": false}
      }
    }
  ],
  "metadata": {
    "domain": "general",
    "overall_preference": 1,
    "individual_preference": [
      {"reasoning": "Response A is better because..."}
    ]
  }
}
```

### Key Fields

- **`input`**: User query in message format
- **`output`**: List of response candidates (typically 2 for pairwise comparison)
- **`preference`**: "chosen" or "rejected" labels
- **`is_preferred`**: Boolean preference indicator
- **`domain`**: Content domain for filtering (e.g., "general", "multilingual", "math")
- **`overall_preference`**: Numeric preference (-1, 0, 1)
- **`individual_preference`**: Optional reasoning for preferences

### Data Loading & Conversion

For loading and converting data from various sources (HuggingFace datasets, local files, etc.), we provide a unified data loading framework. See the **[Data Loading Tutorial](../data/load.ipynb)** for comprehensive examples.

**Quick Example - Load HelpSteer3 Preference Dataset:**

```python
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.build import create_builder
import rm_gallery.core.data
import rm_gallery.gallery.data

# Load HelpSteer3 preference data
config = {
    "path": "HelpSteer3/preference/train.jsonl",
    "limit": 1000
}

load_module = create_loader(
    name="helpsteer3_train",
    load_strategy_type="local",
    data_source="helpsteer3_preference",  # Uses HelpSteer3PreferenceConverter
    config=config
)

pipeline = create_builder(
    name="load_pipeline",
    load_module=load_module
)

result = pipeline.run()
print(f"Loaded {len(result)} samples")

# Each sample contains:
# - Multi-turn conversation input
# - Two response candidates with preference labels
# - Domain and language metadata
# - Overall preference scores (-3 to +3)
```

## 🔧 Advanced Usage

### Checkpoint and Resume

The generation process supports checkpointing for long-running tasks:

```bash
# Enable resume in run_generator.sh
RESUME="--resume"

# Or disable checkpointing for faster processing
DISABLE_CHECKPOINT="--disable-checkpoint"
```

**Checkpoint Files:**
- `checkpoint_samples.jsonl`: Incremental progress save
- Resume automatically skips processed samples
- Safe interruption with Ctrl+C

### Domain-Specific Generation

Filter training data by domain for specialized rubrics:

```bash
# In run_generator.sh, set domain filter
DOMAINS="multilingual"  # or "general", "math", etc.

# Or remove domain filter for all data
# DOMAINS=""
```

### Custom Analysis

Compare different rubric sets:

```bash
# Compare structured vs. raw rubrics
python run_rubric_analysis.py \
    --rubrics rubric_structuring_results/ready_to_use_rubrics.json \
    --source-rubrics rubric_generation_output/rubrics.json \
    --output comparison_analysis
```

## 🔬 Technical Details

### Propose-Evaluate-Revise Loop

1. **Propose**: Generate rubrics using LLM with preference context
2. **Evaluate**: Test rubrics against ground-truth preferences
3. **Revise**: Improve rubrics based on evaluation feedback
4. **Repeat**: Continue until convergence or max epochs

### MCR² Selection Algorithm

Information-theoretic selection maximizes rubric diversity while maintaining quality:
- Selects rubrics that maximize coding rate
- Promotes semantic diversity in rubric set
- Prevents redundant or overlapping criteria

### Theme-Tips Structuring

Hierarchical organization of rubrics:
- **Theme**: High-level evaluation focus
- **Tips**: Specific actionable guidelines
- LLM-based semantic clustering and synthesis

---

**Note**: This framework is designed for research and experimentation. For production deployment, conduct thorough validation on your specific use cases and datasets.
