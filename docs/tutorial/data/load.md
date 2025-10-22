# Data Loading Module

## 1. Overview

The data loading module provides a unified, flexible data loading interface that supports loading data from multiple data sources and converting them to standardized formats. This module is located in the `rm_gallery/core/data/load/` directory.

## 2. Core Architecture

### Design Patterns
- **Strategy Pattern**: Supports different data loading strategies
  - `FileDataLoadStrategy`: Local file loading
  - `HuggingFaceDataLoadStrategy`: HuggingFace dataset loading

- **Registry Pattern**: Dynamic registration and management of data converters
  - `DataConverterRegistry`: Converter registry center
  - Supports runtime registration of new data format converters

- **Template Method Pattern**: Unified data conversion interface
  - `DataConverter`: Abstract converter base class
  - Various concrete converters implement specific format conversion logic

## 3. Supported Data Sources

### Local Files
- **Supported Formats**: JSON (`.json`), JSONL (`.jsonl`), Parquet (`.parquet`)
- **Core Features**:
  - Automatic file type detection
  - Batch file loading
  - Recursive directory scanning

### Hugging Face Datasets
- **Data Source**: Hugging Face Hub public datasets
- **Core Features**:
  - Streaming data loading
  - Flexible configuration options
  - Support for dataset sharding

## 4. Built-in Data Converters

### ChatMessageConverter (`chat_message`)
Specifically handles chat conversation format data:
```python
{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]
}
```

### GenericConverter (`*`)
Generic converter that automatically recognizes common fields:
```python
{
    "prompt": "User input",      # Supported fields: question, input, text, instruction
    "response": "Model reply"    # Supported fields: answer, output, completion
}
```

### Supported Benchmark Datasets

Currently built-in support for converters for the following benchmark datasets (located in `rm_gallery/gallery/data/load/`):

- **rewardbench**
- **rewardbench2**
- **helpsteer2**
- **prmbench**
- **rmbbenchmark_bestofn**
- **rmbbenchmark_pairwise**

Each dataset has a corresponding dedicated converter that can correctly handle its specific data format and field structure.

## 5. Quick Start

### Local File Loading



```python
# Implementation by creating factory function
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.build import create_builder
import rm_gallery.core.data     # Core strategy registration
import rm_gallery.gallery.data  # Extended strategy registration

config = {
    "path": "../../../data/reward-bench-2/data/test-00000-of-00001.parquet",
    "limit": 1000,  # Limit the number of data items to load
}

# Create loading module
load_module = create_loader(
    name="rewardbench2",
    load_strategy_type="local",
    data_source="rewardbench2",
    config=config
)
# Create complete pipeline
pipeline = create_builder(
    name="load_pipeline",
    load_module=load_module
)

# Run pipeline
result = pipeline.run()
print(f"Successfully loaded {len(result)} data items")

```

    Successfully loaded 1000 data items


### Hugging Face Dataset Loading



```python
# Implementation by creating factory function
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.build import create_builder

config = {
    "huggingface_split": "test",        # Dataset split (train/test/validation)
    "limit": 1000,          # Limit the number of data items to load
    "streaming": False      # Whether to use streaming loading
}

# Create loading module
load_module = create_loader(
    name="allenai/reward-bench-2",
    load_strategy_type="huggingface",
    data_source="rewardbench",
    config=config
)
# Create complete pipeline
pipeline = create_builder(
    name="load_pipeline",
    load_module=load_module
)

# Run pipeline
result = pipeline.run()
print(f"Successfully loaded {len(result)} data items")

```

### Data Export

Built-in data export capabilities supporting multiple format data export: jsonl, parquet, json, and splitting into training and test sets.



```python
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.build import create_builder
from rm_gallery.core.data.export import create_exporter
import rm_gallery.core.data     # Core strategy registration
import rm_gallery.gallery.data  # Extended strategy registration


config = {
    "path": "../../../data/reward-bench-2/data/test-00000-of-00001.parquet",
    "limit": 1000,  # Limit the number of data items to load
}

# Create loading module
load_module = create_loader(
    name="rewardbench2",
    load_strategy_type="local",
    data_source="rewardbench2",
    config=config
)

export_module = create_exporter(
    name="rewardbench2",
    config={
        "output_dir": "./exports",
        "formats": ["jsonl"],
        "split_ratio": {"train": 0.8, "test": 0.2}
    }
)
# Create complete pipeline
pipeline = create_builder(
    name="load_pipeline",
    load_module=load_module,
    export_module=export_module
)

# Run pipeline
result = pipeline.run()
print(f"Successfully loaded {len(result)} data items")

```

    [32m2025-07-02 12:26:34.230[0m | [1mINFO    [0m | [36mrm_gallery.core.data.build[0m:[36mrun[0m:[36m85[0m - [1mStarting data build pipeline: load_pipeline[0m
    [32m2025-07-02 12:26:34.232[0m | [1mINFO    [0m | [36mrm_gallery.core.data.build[0m:[36mrun[0m:[36m97[0m - [1mStage: Loading[0m
    [32m2025-07-02 12:26:34.669[0m | [1mINFO    [0m | [36mrm_gallery.core.data.load.base[0m:[36m_load_data_impl[0m:[36m392[0m - [1mLoaded 1865 samples from file: ../../../data/reward-bench-2/data/test-00000-of-00001.parquet[0m
    [32m2025-07-02 12:26:34.670[0m | [1mINFO    [0m | [36mrm_gallery.core.data.load.base[0m:[36mrun[0m:[36m262[0m - [1mApplied limit of 1000, final count: 1000[0m
    [32m2025-07-02 12:26:34.670[0m | [1mINFO    [0m | [36mrm_gallery.core.data.load.base[0m:[36mrun[0m:[36m276[0m - [1mSuccessfully loaded 1000 items from rewardbench2[0m
    [32m2025-07-02 12:26:34.673[0m | [1mINFO    [0m | [36mrm_gallery.core.data.build[0m:[36mrun[0m:[36m99[0m - [1mLoading completed: 1000 items[0m
    [32m2025-07-02 12:26:34.674[0m | [1mINFO    [0m | [36mrm_gallery.core.data.build[0m:[36mrun[0m:[36m97[0m - [1mStage: Export[0m
    [32m2025-07-02 12:26:34.675[0m | [1mINFO    [0m | [36mrm_gallery.core.data.export[0m:[36m_split_dataset[0m:[36m381[0m - [1mIndividual split: 800 training samples, 200 test samples[0m
    [32m2025-07-02 12:26:34.859[0m | [1mINFO    [0m | [36mrm_gallery.core.data.export[0m:[36m_export_jsonl[0m:[36m452[0m - [1mExported to JSONL: exports/rewardbench2_train.jsonl[0m
    [32m2025-07-02 12:26:34.908[0m | [1mINFO    [0m | [36mrm_gallery.core.data.export[0m:[36m_export_jsonl[0m:[36m452[0m - [1mExported to JSONL: exports/rewardbench2_test.jsonl[0m
    [32m2025-07-02 12:26:34.908[0m | [1mINFO    [0m | [36mrm_gallery.core.data.export[0m:[36mrun[0m:[36m138[0m - [1mSuccessfully exported 1000 samples to exports[0m
    [32m2025-07-02 12:26:34.908[0m | [1mINFO    [0m | [36mrm_gallery.core.data.build[0m:[36mrun[0m:[36m99[0m - [1mExport completed: 1000 items[0m
    [32m2025-07-02 12:26:34.909[0m | [1mINFO    [0m | [36mrm_gallery.core.data.build[0m:[36mrun[0m:[36m101[0m - [1mPipeline completed: 1000 items processed[0m


    Successfully loaded 1000 data items


## 6. Data Output Format

### BaseDataSet Structure
All loaded data is encapsulated as a `BaseDataSet` object:
```python
BaseDataSet(
    name="dataset_name",           # Dataset name
    metadata={                     # Metadata information
        "source": "data_source",
        "strategy_type": "local|huggingface",
        "config": {...}
    },
    datasamples=[DataSample(...), ...]   # List of standardized data samples
)
```

### DataSample Structure
Each data sample is uniformly converted to `DataSample` format:
```python
DataSample(
    unique_id="md5_hash_id",        # Unique identifier for the data
    input=[                         # Input message list
        ChatMessage(role="user", content="...")
    ],
    output=[                        # Output data list
        DataOutput(answer=Step(...))
    ],
    source="data_source_name",      # Data source name
    task_category="chat|qa|instruction_following|general",  # Task category
    metadata={                      # Detailed metadata
        "raw_data": {...},          # Raw data
        "load_strategy": "ConverterName",  # Converter used
        "source_file_path": "...",  # Source file path (local files)
        "dataset_name": "...",      # Dataset name (HF datasets)
        "load_type": "local|huggingface"   # Loading method
    }
)
```

## 7. Custom Data Converters

If you need to support new data formats, you can create custom converters by following these steps:

### Step 1: Implement Converter Class
Create a converter file in the `rm_gallery/gallery/data/load/` directory:

```python
from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry

@DataConverterRegistry.register("custom_format")
class CustomConverter(DataConverter):
    """Custom data format converter"""

    def convert_to_data_sample(self, data_dict, source_info):
        """
        Convert raw data to DataSample format

        Args:
            data_dict: Raw data dictionary
            source_info: Data source information

        Returns:
            DataSample: Standardized data sample
        """
        # Implement specific conversion logic
        return DataSample(...)
```

### Step 2: Register Converter
Import the converter in `rm_gallery/gallery/data/__init__.py` to complete registration:

```python
from rm_gallery.gallery.data.load.custom_format import CustomConverter
```

