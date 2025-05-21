# RM-Gallery Data Processing System

## Overview

The RM-Gallery data processing system is designed to handle the loading, processing, and transformation of evaluation data for reinforcement learning models. The system is built with flexibility and extensibility in mind, allowing for easy addition of new data sources and processing steps.

## System Architecture

### Core Components

1. **Base Data Structure (`base.py`)**
   - Defines the base data classes using Pydantic models
   - Includes `BaseData` and `BaseDataSet` classes
   - Provides common functionality for data management

2. **Data Schema (`data_schema.py`)**
   - Defines the core data structures using Pydantic models
   - Includes `ContentList`, `ContextList`, `Reward`, `DataInfo`, and `EvaluationSample`
   - Ensures data validation and type safety

3. **Data Load Strategy (`data_load_strategy.py`)**
   - Implements the Strategy pattern for data loading
   - Supports multiple data sources (local files, Huggingface)
   - Uses a registry system with wildcard matching for strategy management
   - Includes built-in strategies for JSON, JSONL, and Parquet files

4. **Data Processor (`data_processor.py`)**
   - Implements the Pipeline pattern for data processing
   - Provides a framework for adding custom operators
   - Includes built-in operators for filtering, mapping, and reward processing
   - Supports integration with data-juicer operators

5. **Data Builder (`data_builder.py`)**
   - Handles the construction of evaluation datasets
   - Manages data loading and saving operations
   - Provides a clean interface for data manipulation

### Design Patterns

1. **Strategy Pattern**
   - Used for data loading strategies
   - Allows easy addition of new data sources
   - Provides flexible strategy selection with wildcard matching

2. **Pipeline Pattern**
   - Used for data processing
   - Enables modular processing steps through operators
   - Supports both item-level and dataset-level processing

3. **Registry Pattern**
   - Used for strategy management
   - Provides flexible strategy lookup with wildcard support
   - Enables dynamic strategy registration

## Usage Examples

### Basic Usage

```python
from rm_gallery.data import DataPipeline, OperatorFactory

# Create pipeline
pipeline = DataPipeline(name="my_pipeline")

# Add operators
pipeline.add_operator(OperatorFactory.create_operator({
    'type': 'filter',
    'name': 'length_filter',
    'item_filter_func': 'lambda x: len(x.evaluation_sample.input) > 0'
}))

pipeline.add_operator(OperatorFactory.create_operator({
    'type': 'data_juicer',
    'name': 'perplexity_filter',
    'config': {
        'lang': 'en',
        'max_ppl': 1000
    }
}))

# Run pipeline
processed_data = pipeline.run(data_items)
```

### Custom Operator

```python
from rm_gallery.data import Operator

class CustomOperator(Operator):
    def process_item(self, item):
        # Custom item-level processing
        return item

    def process_dataset(self, items):
        # Custom dataset-level processing
        return items

# Add to pipeline
pipeline.add_operator(CustomOperator(name="custom_op"))
```

### Loading from Different Sources

```python
from rm_gallery.data import DataLoadStrategyRegistry

# Load from local JSON file
strategy = DataLoadStrategyRegistry.get_strategy_class(
    data_type='local',
    data_source='rewardbench',
    dimension='helpfulness'
)
data = strategy(config={'path': 'path/to/data.json'}).load_data()

# Load from Huggingface
strategy = DataLoadStrategyRegistry.get_strategy_class(
    data_type='remote',
    data_source='huggingface',
    dimension='*'
)
data = strategy(config={'dataset_name': 'your-dataset'}).load_data()
```

## Extending the System

### Adding a New Data Source

1. Create a new strategy class:
```python
from rm_gallery.data import DataLoadStrategy, DataLoadStrategyRegistry

@DataLoadStrategyRegistry.register('remote', 'new_source', '*')
class NewSourceDataLoadStrategy(DataLoadStrategy):
    def validate_config(self, config):
        # Validate config
        pass

    def load_data(self, **kwargs):
        # Load data
        pass
```

### Adding a New Operator

1. Create a new operator class:
```python
from rm_gallery.data import Operator

class NewOperator(Operator):
    def process_item(self, item):
        # Process single item
        return item

    def process_dataset(self, items):
        # Process entire dataset
        return items
```

## Best Practices

1. **Error Handling**
   - Use try-except blocks in operators
   - Log errors using loguru
   - Provide meaningful error messages

2. **Configuration**
   - Validate configurations in strategy classes
   - Provide default values where appropriate
   - Document configuration options

3. **Performance**
   - Use batch processing when possible
   - Implement efficient data loading strategies
   - Consider memory usage for large datasets

4. **Testing**
   - Write unit tests for operators
   - Test with different data sources
   - Include edge cases in tests

## Dependencies

- pydantic
- loguru
- pandas (for Parquet support)
- data-juicer (for advanced data processing) 