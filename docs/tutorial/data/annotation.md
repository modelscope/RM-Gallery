# Data Annotation Module

The Data Annotation Module is built on Label Studio, providing efficient and flexible data annotation solutions for machine learning projects. It supports multiple annotation scenarios and is particularly suitable for data preparation in reward model and dialogue system projects.

## 1. Overview

- **Deep Label Studio Integration**: Uses Label Studio as the annotation interface, providing an intuitive and user-friendly annotation experience
- **Multi-scenario Data Support**: Comprehensive support for dialogue annotation, quality scoring, preference ranking, and other annotation tasks
- **Quick Deployment**: Provides both Docker and pip deployment options with one-click service startup
- **Templated Annotation Configuration**: Built-in annotation templates like RewardBench for out-of-the-box usage
- **Seamless Ecosystem Integration**: Deep integration with RM Gallery data processing pipeline for smooth data flow
- **Enterprise-level Batch Processing**: Supports large-scale batch annotation, export, and management

## 2. Application Scenarios

This module is primarily suitable for the following machine learning data preparation scenarios:

1. **Reward Model Training Data Annotation** - Preparing high-quality preference data for reward model training
2. **Dialogue System Quality Assessment** - Evaluating and improving the output quality of dialogue models
3. **Preference Learning Data Preparation** - Building comparison datasets for preference learning
4. **Text Classification and Sentiment Analysis** - Preparing annotated data for supervised learning tasks

## 3. Quick Start

### 3.1. Environment Setup

Ensure the following dependencies are installed:
```
label_studio==1.17.0
```

### 3.2. Start Label Studio Annotation Service

Use the following commands to start the annotation service:

```bash
# Start using Docker (recommended)
python ./rm_gallery/core/data/annotation/server.py start

# Start using pip
python ./rm_gallery/core/data/annotation/server.py start --use-pip

# Check service status
python ./rm_gallery/core/data/annotation/server.py status

# Stop annotation service
python ./rm_gallery/core/data/annotation/server.py stop
```

After successful startup, the console will display:
```
============================================================
🚀 Label Studio Successfully Started!
============================================================
🌐 Web Interface: http://localhost:8080
📧 Username: admin@rmgallery.com
🔐 Password: RM-Gallery
📁 Data Directory: ./log/label_studio_logs
🐳 Deployment: Pip
============================================================
```

### 3.3. Verify Service Status

Run the status check command to confirm the service is running normally:
```
==================================================
📊 Label Studio Status
==================================================
🌐 Server URL: http://localhost:8080
🚀 Deployment: PIP
🔌 Port: 8080
✅ Running
📁 Data Dir: ./log/label_studio_logs
👤 Username: admin@rmgallery.com
🔄 Process PIDs: 65727
🔌 Port PIDs: 65727
==================================================
```

### 3.4. Obtain API Token

After completing service startup, follow these steps to obtain the API Token:

1. Visit http://localhost:8080 in your browser
2. Login using the following credentials:
   - Username: `admin@rmgallery.com`
   - Password: `RM-Gallery`
3. Click "Organization" in the left navigation bar and go to API Tokens Settings
4. Set both Personal Access Tokens and Legacy Tokens to True
5. Click the user avatar in the top right corner and select "Account & Settings"
6. Copy the token value from the Access Token section - this is the API Token you'll need later


## 4. Complete Usage Example

The following complete example demonstrates how to use the data annotation module, including the full workflow of data import, project creation, annotation execution, and result export.



```python
"""
Step 1: Import data and create annotation project
"""

from rm_gallery.core.data.annotation.annotation import create_annotator
from rm_gallery.core.data.load.base import create_loader
import rm_gallery.core.data     # Core strategy registration
import rm_gallery.gallery.data  # Extension strategy registration

# Replace with your actual API Token obtained from Label Studio
API_TOKEN = ""

# Step 1.1: Configure data loading parameters
load_config = {
    "path": "../../../data/reward-bench-2/data/test-00000-of-00001.parquet",  # Replace with actual data path
    "limit": 1000,  # Limit the number of records loaded to avoid loading too much data for initial testing
}

# Step 1.2: Create data loader
loader = create_loader(
    name="rewardbench2",           # Dataset identifier name
    load_strategy_type="local",    # Use local file loading strategy
    data_source="rewardbench2",    # Specify data source format converter
    config=load_config             # Pass loading configuration parameters
)

# Step 1.3: Execute data loading
print("Loading data...")
dataset = loader.run()
print(f"Data loading completed, {len(dataset.datasamples)} records total")

# Step 1.4: Configure annotation project parameters
annotation_config = {
    "server_url": "http://localhost:8080",           # Label Studio service address
    "api_token": API_TOKEN,                          # API access token
    "project_title": "RM Gallery Quality Annotation", # Project display name
    "template_name": "rewardbench2",                 # Use built-in RewardBench2 template
    "project_description": "Data quality annotation project based on RewardBench2 template"
}

# Step 1.5: Create annotation module instance
annotation_module = create_annotator(
    name="rm_gallery_annotation",
    **annotation_config
)

# Step 1.6: Create annotation project and import data
print("Creating annotation project...")
result = annotation_module.run(dataset, create_new_project=True)

if result:
    project_url = f"{result.metadata['annotation_server_url']}/projects/{result.metadata['annotation_project_id']}"
    print(f"✅ Annotation project created successfully!")
    print(f"🌐 Project URL: {project_url}")
    print(f"📊 Project ID: {result.metadata['annotation_project_id']}")
else:
    print("❌ Failed to create annotation project, please check configuration and network connection")

```


```python
"""
Step 2: Perform data annotation

After the project is created successfully, you can:
1. Visit the project URL output above
2. Login using admin@rmgallery.com / RM-Gallery
3. Annotate data in the annotation interface
4. After annotation is complete, run the next step to export annotation results

Note: In actual usage, you need to manually complete the annotation work, then run the export code below.
"""

# This is a placeholder for annotation operations
# Actual annotation work needs to be completed in the Label Studio web interface
print("📝 Please complete data annotation work in the Label Studio interface")
print("💡 After annotation is complete, run the code below to export results")

```


```python
"""
Step 3: Export annotation results
"""

from rm_gallery.core.data.annotation.annotation import create_annotator
from rm_gallery.core.data.export import create_exporter
import rm_gallery.core.data     # Core strategy registration
import rm_gallery.gallery.data  # Extension strategy registration

# Use the same API Token as when creating the project
API_TOKEN = ""

# Step 3.1: Recreate annotation module instance
annotation_module = create_annotator(
    name="rm_gallery_annotation",
    template_name="rewardbench2",
    api_token=API_TOKEN
)

# Step 3.2: Set project ID (obtained from Step 1 output)
annotation_module.project_id = 3  # Replace with actual project_id

# Step 3.3: Export annotation data from Label Studio
print("Exporting annotation data...")
try:
    annotated_dataset = annotation_module.export_annotations_to_dataset()
    print(f"✅ Annotation data exported successfully, {len(annotated_dataset.datasamples)} annotated records total")
except Exception as e:
    print(f"❌ Export failed: {e}")
    annotated_dataset = None

# Step 3.4: Configure file exporter
if annotated_dataset:
    exporter = create_exporter(
        name="annotation_exporter",
        config={
            "output_dir": "./exports",          # Export file storage directory
            "formats": ["jsonl"]        # Support multiple export formats
        }
    )

    # Step 3.5: Execute data export
    print("Saving annotation results to file...")
    export_result = exporter.run(annotated_dataset)

    if export_result:
        print("✅ Annotation results saved to ./exports directory")
        print(f"📊 Dataset info: {annotated_dataset.name}")
        print(f"📁 Contains {len(annotated_dataset.datasamples)} annotation data")
    else:
        print("❌ File export failed")

    # Display partial data preview
    if annotated_dataset.datasamples:
        print("\n📋 Data preview:")
        sample = annotated_dataset.datasamples[0]
        print(f"  - Sample ID: {sample.unique_id}")
        print(f"  - Annotation status: {sample.metadata.get('annotation_status', 'unknown')}")
        print(f"  - Output count: {len(sample.output) if sample.output else 0}")

```

## 5. Built-in Annotation Templates

The system provides the following pre-configured annotation templates for out-of-the-box usage, located in `rm_gallery/gallery/data/annotation/`:

| Template Name | Template ID | Source | Description |
|---------------|-------------|--------|-------------|
| RewardBenchAnnotationTemplate | `rewardbench` | RewardBench | Supports 2-choice quality scoring and ranking |
| RewardBench2AnnotationTemplate | `rewardbench2` | RewardBench2 | Supports 4-choice quality scoring and ranking |

## 6. Custom Annotation Template Development

If the built-in templates don't meet your annotation needs, you can develop custom templates following these steps:

### Step 1: Create Template Class

Create a new template file in the `rm_gallery/gallery/data/annotation/` directory, inheriting from the `BaseAnnotationTemplate` base class:

```python
from rm_gallery.core.data.annotation.template import BaseAnnotationTemplate, AnnotationTemplateRegistry

@AnnotationTemplateRegistry.register("custom_template")
class CustomAnnotationTemplate(BaseAnnotationTemplate):
    @property
    def label_config(self) -> str:
        """
        Define Label Studio annotation interface configuration
        Using Label Studio's XML configuration syntax
        """
        return """
        <View>
            <Text name="question" value="$question"/>
            <Choices name="quality" toName="question" choice="single-radio">
                <Choice value="excellent" background="green"/>
                <Choice value="good" background="blue"/>
                <Choice value="fair" background="yellow"/>
                <Choice value="poor" background="red"/>
            </Choices>
            <Rating name="score" toName="question" maxRating="10" />
            <TextArea name="comments" toName="question"
                     placeholder="Please enter evaluation reason..." rows="3"/>
        </View>
        """

    def process_annotations(self, annotation_data):
        """
        Process annotation data obtained from Label Studio
        Convert raw annotation data to structured format
        """
        processed_data = {
            "quality_rating": annotation_data.get("choices", {}).get("quality", {}).get("choices", []),
            "numerical_score": annotation_data.get("rating", {}).get("score", {}).get("rating", 0),
            "textual_feedback": annotation_data.get("textarea", {}).get("comments", {}).get("text", [""])[0]
        }
        return processed_data
```

### Step 2: Register Template

Import your template class in the `rm_gallery/gallery/data/__init__.py` file to complete registration:

```python
# Import custom annotation template
from rm_gallery.gallery.data.annotation.custom_template import CustomAnnotationTemplate
```

### Step 3: Use Custom Template

When creating an annotation module, specify your custom template name:

```python
annotation_module = create_annotation_module(
    name="custom_annotation_project",
    template_name="custom_template",  # Use your registered template name
    # ... other configuration parameters
)
```

## 7. Data Format Specifications

### Input Data Requirements

- **DataSample Standard Format**

### Output Data Format

- **Annotation Result Integration**: Annotation data is automatically added to the `label` field of DataSample
- **Original Data Protection**: Maintains integrity and original structure of input data
- **Rich Metadata**: Includes annotation project ID, annotation status, timestamps, and other tracking information
- **Multi-format Export**: Supports JSON, JSONL, and other export formats

### Data Structure Example

```python
# Annotated DataSample structure example
data_sample = DataSample(
    unique_id="sample_001",
    input=[...],  # Original input data
    output=[...], # Original output data (if any)
    metadata={
        "annotation_status": "completed",
        "annotation_project_id": 123,
        "annotator_id": "user@example.com"
    }
)

# The label field in each output contains annotation results
output.label = {
    "annotation_data": {
        "ratings": {...},      # Rating data
        "choices": {...},      # Choice data
        "text_areas": {...}    # Text input data
    },
    "processed": {...}         # Structured data processed by template
}
```

## 8. Troubleshooting Guide

### 8.1. Common Issues and Solutions

#### 8.1.1. Label Studio Service Startup Failure

**Problem**: Service cannot start or is inaccessible after startup

**Solution Steps**:
```bash
# Check if port is occupied
lsof -i :8080

# If port is occupied, start with different port
python ./rm_gallery/core/data/annotation/server.py start --port 8081

# View detailed startup logs
python ./rm_gallery/core/data/annotation/server.py start --data-dir ./custom_log --verbose

# Clean previous data directory (use with caution)
rm -rf ./log/label_studio_logs
```

#### 8.1.2. API Token Acquisition Failure

**Problem**: Cannot find API Token in the interface

**Solution Steps**:
1. Ensure correct login to Label Studio interface
2. Check user permission settings, ensure admin privileges
3. Enable API Tokens feature in Organization settings
4. If still unable to obtain, try recreating user account

#### 8.1.3. Data Import Failure

**Problem**: Project created successfully but data cannot be imported

**Solution Steps**:
- Check if data format meets requirements
- Verify API Token correctness
- Confirm normal network connection
- Check if data size exceeds limits

#### 8.1.4. Annotation Result Export Exception

**Problem**: Exported data is incomplete or format is abnormal

**Solution Steps**:
- Confirm all data has been annotated
- Check write permissions for export path
- Verify correct implementation of template's process_annotations method

### 8.2 Getting Technical Support

If you encounter unresolvable issues, you can get help through the following methods:

1. **View log files**: Log files in the `./log/label_studio_logs/` directory
2. **Check system status**: Run `python ./rm_gallery/core/data/annotation/server.py status`
3. **Restart service**: Completely stop the service and restart
4. **Community support**: Submit an Issue in the project repository with detailed error information and reproduction steps

