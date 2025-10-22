# Data Module

## 1. Overview

The Data Module provides a complete data processing solution covering the entire lifecycle from data loading, preprocessing, quality annotation to export. This module supports multiple operation modes and can flexibly combine different data processing components to meet various data processing scenario requirements.

## 2. System Architecture

The Data Module adopts a modular design consisting of five core components:

### 2.1. Load Module
   - Supports local files and remote data sources (such as HuggingFace Hub)
   - Supports multiple data formats: parquet, jsonl, json, etc.
   - Built-in data source adapters: rewardbench, chatmessage, prmbench, etc.
   - Supports data splitting and sampling limits

### 2.2. Process Module
   - Configurable data processing pipeline
   - Built-in filters: text length filtering, conversation turn filtering, etc.
   - Integrated data-juicer advanced data cleaning operators
   - Supports custom processor extensions

### 2.3. Annotation Module
   - Deep integration with Label Studio annotation platform
   - Supports multiple preset annotation templates
   - Automatic project creation and configuration management
   - Supports multi-user collaborative annotation

### 2.4. Export Module
   - Multi-format data export: jsonl, parquet, json
   - Intelligent data splitting (train/test sets)
   - Maintains original data directory structure

### 2.5. Build Module
   - Unified data pipeline orchestration
   - Automatic inter-module data flow management
   - Supports YAML configuration-based building
   - Supports pipeline reuse and extension

## 3. Operation Modes

The Data Module supports two main operation methods: **Python Script Mode** and **YAML Configuration Mode**, catering to different user preferences.

### 3.1. Python Script Mode (data_pipeline.py)

For the complete pipeline script, please refer to `./examples/data/data_pipeline.py`

#### 3.1.1. Basic Data Processing Flow
Execute the complete data processing pipeline: **Data Loading → Data Processing → Data Export**

```bash
# Process 100 sample data points
python data_pipeline.py --mode basic --limit 100
```

#### 3.1.2. Complete Flow with Annotation
Execute the complete pipeline including manual annotation: **Data Loading → Data Processing → Data Annotation → Data Export**

```bash
# Requires Label Studio API Token
python data_pipeline.py --mode annotation --api-token YOUR_LABEL_STUDIO_TOKEN
```

#### 3.1.3. Independent Module Testing Mode
Supports testing individual module functionality:

- **Load Only**: `python data_pipeline.py --mode load-only`
- **Process Only**: `python data_pipeline.py --mode process-only`
- **Export Only**: `python data_pipeline.py --mode export-only`

#### 3.1.4. Annotation Data Export Mode
Export completed annotation data from Label Studio:

```bash
python data_pipeline.py --mode export-annotation \
    --api-token YOUR_TOKEN \
    --project-id PROJECT_ID
```

### 3.2 YAML Configuration Mode (data_from_yaml.py)

Run data pipelines through declarative YAML configuration files, more suitable for batch processing and production environments:

```bash
python data_from_yaml.py --config ./examples/data/config.yaml
```


## 4. Configuration File Details

### YAML Configuration File Structure

The YAML configuration file provides a declarative pipeline configuration approach, supporting complete data processing flow definition. Here's the complete configuration file structure explanation:

```yaml
dataset:
    # Dataset basic information
    name: rewardbench2                    # Dataset name
                                          # local mode: custom name (e.g., rewardbench2)
                                          # huggingface mode: HF dataset name (e.g., allenai/reward-bench-2)

    # Data source configuration
    configs:
        type: local                       # Data source type
                                          # - local: Local file system
                                          # - huggingface: HuggingFace Hub
        source: rewardbench2              # Data source adapter identifier
                                          # Note: Ensure corresponding converter is registered
        path: /path/to/data.parquet       # Data file path (local mode only)
        huggingface_split: train          # Data split name (huggingface mode only)
                                          # Options: train, test, validation, etc.
        limit: 2000                       # Sample count limit (random sampling)
                                          # Used for quick testing or data preview

    # Data processor configuration (optional)
    processors:
        # Conversation turn filter
        - type: filter
          name: conversation_turn_filter
          config:
            min_turns: 1
            max_turns: 6

        # Text length filter
        - type: filter
          name: text_length_filter
          config:
            min_length: 10
            max_length: 1000

        # data-juicer operator example
        - type: data_juicer
          name: character_repetition_filter
          config:
            rep_len: 10
            min_ratio: 0.0
            max_ratio: 0.5

    # Annotation configuration (optional)
    annotation:
        template_name: "rewardbench2"     # Annotation template name
        project_title: "Reward Bench Evaluation"  # Label Studio project title
        project_description: "Reward model evaluation using reward bench template from yaml"
        server_url: "http://localhost:8080"        # Label Studio server address
        api_token: "your_api_token_here"          # Label Studio API token

    # Export configuration (required)
    export:
        output_dir: ./examples/data/exports       # Export directory path
        formats: ["jsonl"]                        # Export format list
                                                  # Supported: jsonl, parquet, json
        preserve_structure: true                  # Whether to maintain original directory structure
        split_ratio: {"train": 0.8, "test": 0.2} # Dataset split ratio
                                                  # Supports multiple splits: train/test, comment out if no splitting needed

    # Metadata configuration (optional)
    metadata:
        source: "rewardbench2"            # Data source identifier
        version: "1.0"                    # Data version (optional)
        description: "Sample dataset"      # Data description (optional)
```


## 5. Reference Resources

### Official Documentation
- **Label Studio Official Guide**: https://labelstud.io/guide/
- **Data-Juicer Project Documentation**: https://github.com/modelscope/data-juicer
- **HuggingFace Datasets**: https://huggingface.co/docs/datasets/

