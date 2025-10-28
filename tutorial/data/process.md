# Data Process Module

## 1. Overview

The Data Process Module provides users with a unified and flexible data processing solution. Based on the **Operator Pipeline** design philosophy, this module allows users to build complex data processing workflows by flexibly combining multiple operators.

## 2. Architecture Design

### Core Components

#### 2.1. DataProcess - Data Processing Engine
   - Inherits from `BaseDataModule`, providing standardized data processing interfaces
   - Manages and orchestrates the execution order of operator sequences
   - Supports both batch data processing and real-time data stream processing

#### 2.2. BaseOperator - Abstract Base Class for Operators
   - Defines standard interface specifications for operators
   - Supports generic types for type safety
   - Provides extensible data processing abstract methods

#### 2.3. OperatorFactory - Operator Factory
   - Implements unified registration and dynamic creation mechanisms for operators
   - Seamlessly integrates with the data-juicer ecosystem operators
   - Supports configuration-based operator instantiation

## 3. Core Features

### 3.1. Pipeline-based Data Processing
- **Chain Operations**: Supports seamless serial execution of multiple operators
- **Metadata Preservation**: Completely preserves metadata information from original datasets
- **Full Tracking**: Provides detailed processing logs, performance statistics, and data flow tracking

### 3.2. Rich Operator Ecosystem
- **Built-in Operators**:
  - `TextLengthFilter` - Intelligent filter based on text length
  - `ConversationTurnFilter` - Filter for conversation turn count
- **External Integration**:
  - Full support for data-juicer operator library
  - Support for custom operator extensions

### 3.3. Configuration-driven Design
- **Declarative Configuration**: Flexibly define data processing flows through configuration files
- **Parameterized Control**: All operator parameters can be adjusted through configuration files
- **Dynamic Adjustment**: Supports runtime dynamic modification of processing parameters



## 4. Quick Start

### Method 1: Direct Operator Creation



```python
from rm_gallery.core.data.process.process import create_processor
from rm_gallery.core.data.process.ops.filter.text_length_filter import TextLengthFilter
from rm_gallery.core.data.process.ops.filter.conversation_turn_filter import ConversationTurnFilter
from rm_gallery.core.data.load.base import create_loader
import rm_gallery.core.data     # Core strategy registration
import rm_gallery.gallery.data  # Extension strategy registration

# Configure local file loading parameters
config = {
    "path": "../../../data/reward-bench-2/data/test-00000-of-00001.parquet",
    "limit": 1000,  # Limit the number of data entries to load
}

# Create data loader
loader = create_loader(
    name="rewardbench2",           # Dataset name
    load_strategy_type="local",    # Use local file loading strategy
    data_source="rewardbench2",    # Specify data source format converter
    config=config                  # Pass configuration parameters
)

# Execute data loading
dataset = loader.run()

# Create operators
text_filter = TextLengthFilter(
    name="text_length_filter",
    config={"min_length": 50, "max_length": 2000}
)

turn_filter = ConversationTurnFilter(
    name="conversation_turn_filter",
    config={"min_turns": 1, "max_turns": 10}
)

# Create data processing module
processor = create_processor(
    name="data_processor",
    operators=[text_filter, turn_filter]
)

# Process data
result = processor.run(dataset)
print(f"Before processing: {len(dataset.datasamples)} data entries")
print(f"After processing: {len(result.datasamples)} data entries")

```

    /Users/xielipeng/Library/Caches/pypoetry/virtualenvs/rm-gallery-VQCvXsd2-py3.10/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    [32m2025-07-02 13:24:02.337[0m | [1mINFO    [0m | [36mrm_gallery.core.utils.logger[0m:[36minit_logger[0m:[36m16[0m - [1mstart![0m
    Before processing: 1000 data entries[32m2025-07-02 13:24:02.722[0m | [1mINFO    [0m | [36mrm_gallery.core.data.load.base[0m:[36m_load_data_impl[0m:[36m392[0m - [1mLoaded 1865 samples from file: ../../../data/reward-bench-2/data/test-00000-of-00001.parquet[0m

    After processing: 168 data entries
    [32m2025-07-02 13:24:02.723[0m | [1mINFO    [0m | [36mrm_gallery.core.data.load.base[0m:[36mrun[0m:[36m262[0m - [1mApplied limit of 1000, final count: 1000[0m
    [32m2025-07-02 13:24:02.724[0m | [1mINFO    [0m | [36mrm_gallery.core.data.load.base[0m:[36mrun[0m:[36m276[0m - [1mSuccessfully loaded 1000 items from rewardbench2[0m
    [32m2025-07-02 13:24:02.729[0m | [1mINFO    [0m | [36mrm_gallery.core.data.process.process[0m:[36mrun[0m:[36m92[0m - [1mProcessing 1000 items with 2 operators[0m
    [32m2025-07-02 13:24:02.729[0m | [1mINFO    [0m | [36mrm_gallery.core.data.process.process[0m:[36mrun[0m:[36m99[0m - [1mApplying operator 1/2: text_length_filter[0m
    [32m2025-07-02 13:24:02.734[0m | [1mINFO    [0m | [36mrm_gallery.core.data.process.process[0m:[36mrun[0m:[36m103[0m - [1mOperator text_length_filter completed: 168 items remaining[0m
    [32m2025-07-02 13:24:02.734[0m | [1mINFO    [0m | [36mrm_gallery.core.data.process.process[0m:[36mrun[0m:[36m99[0m - [1mApplying operator 2/2: conversation_turn_filter[0m


    [32m2025-07-02 13:24:02.734[0m | [1mINFO    [0m | [36mrm_gallery.core.data.process.process[0m:[36mrun[0m:[36m103[0m - [1mOperator conversation_turn_filter completed: 168 items remaining[0m
    [32m2025-07-02 13:24:02.734[0m | [1mINFO    [0m | [36mrm_gallery.core.data.process.process[0m:[36mrun[0m:[36m127[0m - [1mProcessing completed: 1000 -> 168 items[0m


### Method 2: Configuration-based Batch Processing

Using configuration files provides more flexible definition of data processing workflows, especially suitable for complex multi-step processing scenarios.



```python
# Create operators through configuration
from rm_gallery.core.data.process.process import create_processor
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.process.ops.base import OperatorFactory
import rm_gallery.core.data     # Core strategy registration
import rm_gallery.gallery.data  # Extension strategy registration

# Configure local file loading parameters
config = {
    "path": "../../../data/reward-bench-2/data/test-00000-of-00001.parquet",
    "limit": 1000,  # Limit the number of data entries to load
}

# Create data loader
loader = create_loader(
    name="rewardbench2",           # Dataset name
    load_strategy_type="local",    # Use local file loading strategy
    data_source="rewardbench2",    # Specify data source format converter
    config=config                  # Pass configuration parameters
)

# Execute data loading
dataset = loader.run()

# Configure multiple operators
operator_configs = [
    {
        "type": "filter",
        "name": "conversation_turn_filter",
        "config": {"min_turns": 1, "max_turns": 8}
    },
    {
        "type": "filter",
        "name": "text_length_filter",
        "config": {"min_length": 100, "max_length": 2000}
    },
    {
        "type": "data_juicer",
        "name": "character_repetition_filter",
        "config": {
            "rep_len": 10,
            "min_ratio": 0.0,
            "max_ratio": 0.5
        }
    }
]

# Batch create operators
operators = [OperatorFactory.create_operator(config) for config in operator_configs]

# Create processor
processor = create_processor(
    name="batch_processor",
    operators=operators
)

result = processor.run(dataset)
print(f"Before processing: {len(dataset.datasamples)} data entries")
print(f"After processing: {len(result.datasamples)} data entries")

```

## 5. Advanced Features

### 5.1. Custom Operator Development

When built-in operators cannot meet specific requirements, you can easily create custom operators. Here's the complete development workflow:

#### Step 1: Implement Operator Class

Create custom operators in the `rm_gallery/gallery/data/process/` directory:

```python
from rm_gallery.core.data.process.ops.base import BaseOperator, OperatorFactory

@OperatorFactory.register("custom_filter")
class CustomFilter(BaseOperator):
    """Custom data filter example"""

    def process_dataset(self, items):
        """
        Core method for processing datasets

        Args:
            items: List of input data items

        Returns:
            List of filtered data items
        """
        filtered_items = []
        for item in items:
            if self._custom_condition(item):
                filtered_items.append(item)
        return filtered_items

    def _custom_condition(self, item):
        """
        Custom filtering condition

        Args:
            item: Single data item

        Returns:
            bool: Whether to keep this data item
        """
        # Implement your filtering logic here
        return True
```

#### Step 2: Register Operator

Import the operator in `rm_gallery/gallery/data/__init__.py` to complete registration:

```python
from rm_gallery.gallery.data.process.custom_filter import CustomFilter
```

### 5.2. Data-Juicer Operator Integration

RM-Gallery seamlessly integrates with the data-juicer ecosystem, allowing you to use its rich collection of data processing operators:

```python
# Configuration example using data-juicer operators
config = {
    "type": "data_juicer",
    "name": "text_length_filter",
    "config": {
        "min_len": 10,
        "max_len": 20
    }
}

operator = OperatorFactory.create_operator(config)
```

## 6. Supported Operators

### RM-Gallery Built-in Operators

| Operator Name | Functionality | Configuration Parameters |
|---------------|---------------|-------------------------|
| `TextLengthFilter` | Filter data samples based on text length | `min_length`, `max_length` |
| `ConversationTurnFilter` | Filter samples based on conversation turn count | `min_turns`, `max_turns` |

### Data-Juicer Integrated Operators

| Operator Name | Functionality | Status |
|---------------|---------------|--------|
| `text_length_filter` | Text length filtering | ✅ Tested |
| `character_repetition_filter` | Character repetition filtering | ✅ Tested |
| `word_repetition_filter` | Word repetition filtering | 🔄 Testing |

> **Tip**: We continuously add and test new operators, stay tuned for more features!

