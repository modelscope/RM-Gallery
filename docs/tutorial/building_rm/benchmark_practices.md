# Benchmark

## 1. Overview
In this guide, we will show the gallery's pipeline on built-in reward benchmark: [RewardBench2](https://huggingface.co/spaces/allenai/reward-bench) and [RMB Bench](https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark).

## 2. Setup

```python
import sys
import os
sys.path.append("../../..")

os.environ["OPENAI_API_KEY"] = ""
os.environ["BASE_URL"] = ""
```

## 3. RewardBench2

RewardBench2 implements a category-based routing system for specialized reward models. It supports the following categories:
- Safety (toxicity detection)
- Focus (content relevance assessment)
- Math (mathematical reasoning evaluation)
- Factuality (truthfulness verification)
- Precise IF (instruction following capability assessment)
- General helpfulness (default fallback)

```python
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Type
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
# Implementation by creating base class
from rm_gallery.core.reward.base import BaseReward
from rm_gallery.core.reward.composition import RouterComposition
from rm_gallery.core.utils.acc import calc_acc
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward
from rm_gallery.gallery.rm.alignment.harmlessness.safety import SafetyListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.focus import FocusListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.math import MathListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.precise_if import PreciseIFListWiseReward
from rm_gallery.gallery.rm.alignment.honesty.factuality import FactualityListWiseReward

# Configure local file loading parameters
config = {
    "path": "./data/reward-bench-2/data/test-00000-of-00001.parquet",
    "limit": 10,  # Limit the number of data items to load
}

# Create loading module
load_module = create_loader(
    name="rewardbench2",
    load_strategy_type="local",
    data_source="rewardbench2",
    config=config
)

dataset = load_module.run()


# Define router
class RewardBench2Router(RouterComposition):
    rewards: Dict[str, Type[BaseReward]] = {
        "safety": SafetyListWiseReward,
        "focus": FocusListWiseReward,
        "math": MathListWiseReward,
        "factuality": FactualityListWiseReward,
        "precis_if": PreciseIFListWiseReward,
        "general": BaseHelpfulnessListWiseReward,
    }

    def _condition(self, sample: DataSample) -> str:
        # Extract third-level category from path and normalize to lowercase
        # Example: "Safety/Content/Toxicity" -> "toxicity"
        try:
            cond = sample.metadata["raw_data"]["subset"].lower()
        except Exception:
            # Fallback to general reward model when path extraction fails
            cond = "general"

        if cond not in self.rewards:
            cond = "general"
        return cond


# Initialize router
router = RewardBench2Router(
    name="reward-bench-2-router",
    params={
        "llm": OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True),
    }
)

# Process each sample through the appropriate reward model
results = router.evaluate_batch(dataset.datasamples, max_workers=128)

print(f"Processed {len(results)} samples with RewardBench2")
print(f"Accuracy: {calc_acc(results)}")
```

## 3. RMBBench

RMBBench provides task-type specific reward modeling for diverse NLP tasks including:
- Brainstorming quality assessment
- Chat response evaluation
- Classification accuracy scoring
- Code generation quality assessment
- Content generation evaluation
- Open QA and closed QA assessment
- Reasoning capability evaluation
- Text rewriting quality
- Role-playing performance
- Summarization effectiveness
- Translation quality
- General helpfulness (default fallback)

```python
from rm_gallery.core.data.load.base import create_loader
from rm_gallery.gallery.rm.alignment.helpfulness.brainstorming import BrainstormingListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.chat import ChatListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.classification import ClassificationListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.closed_qa import ClosedQAListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.code import CodeListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.generation import GenerationListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.open_qa import OpenQAListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.reasoning import ReasoningListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.rewrite import RewriteListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.role_playing import RolePlayingListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.summarization import SummarizationListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.translation import TranslationListWiseReward

# Configure local file loading parameters
config = {
    "path": "./data/reward-bench-2/data/test-00000-of-00001.parquet",
    "limit": 1000,  # Limit the number of data items to load
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

# Define router
class RMBBenchRouter(RouterComposition):
    rewards: Dict[str, Type[BaseReward]] = {
        "brainstorming": BrainstormingListWiseReward,
        "chat": ChatListWiseReward,
        "classification": ClassificationListWiseReward,
        "closed_qa": ClosedQAListWiseReward,
        "code": CodeListWiseReward,
        "generation": GenerationListWiseReward,
        "open_qa": OpenQAListWiseReward,
        "reasoning": ReasoningListWiseReward,
        "rewrite": RewriteListWiseReward,
        "role_playing": RolePlayingListWiseReward,
        "summarization": SummarizationListWiseReward,
        "translation": TranslationListWiseReward,
        "general": BaseHelpfulnessListWiseReward,
    }

    def _condition(self, sample: DataSample) -> str:
        try:
            cond = sample["meta"]["category_path"].split("/")[-2].lower()

        except Exception:
            # Fallback to general reward model when path extraction fails
            cond = "general"

        if cond not in self.rewards:
            cond = "general"
        return cond


# Initialize router
rmb_router = RMBBenchRouter(
    name="rmb-bench-router",
    params={
        "llm": OpenaiLLM(model="qwen3-235b-a22b", enable_thinking=True),
    }
)

# Process samples with automatic task detection
results = rmb_router.evaluate(dataset.datasamples)

print(f"Processed {len(results)} samples with RewardBench2")
print(f"Accuracy: {calc_acc(results)}")
```

