<div align="center">

<p align="center">
  <img src="./docs/images/logo.svg" alt="RM-Gallery Logo" width="500">
</p>

<h3>A unified platform for building, evaluating, and applying reward models.</h3>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/rm-gallery/)
[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue?logo=pypi)](https://pypi.org/project/rm-gallery/)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=markdown)](https://modelscope.github.io/RM-Gallery/)

[Documentation](https://modelscope.github.io/RM-Gallery/) | [Examples](./examples/) | [中文](./README_zh.md)

</div>

## News

- **2025-10-20** - [Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling](https://arxiv.org/abs/2510.17314) - We released a new paper on learning generalizable reward criteria for robust modeling.
- **2025-10-17** - [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/abs/2510.15514) - We introduced techniques to align judge feedback and improve RL stability.
- **2025-07-09** - Released RM-Gallery v0.1.0 on [PyPI](https://pypi.org/project/rm-gallery/)

## Installation

RM-Gallery requires Python 3.10 or higher (< 3.13).

```bash
pip install rm-gallery
```

Or install from source:

```bash
git clone https://github.com/modelscope/RM-Gallery.git
cd RM-Gallery
pip install .
```

## Quick Start

```python
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.data.schema import DataSample

# Choose from 35+ pre-built reward models
rm = RewardRegistry.get("safety_listwise_reward")

# Evaluate your data
sample = DataSample(...)
result = rm.evaluate(sample)
```

See the [quickstart guide](https://modelscope.github.io/RM-Gallery/quickstart/) for a complete example, or try our [interactive notebooks](./examples/).

## Features

### Pre-built Reward Models

Access 35+ reward models for different domains:

```python
rm = RewardRegistry.get("math_correctness_reward")
rm = RewardRegistry.get("code_quality_reward")
rm = RewardRegistry.get("helpfulness_listwise_reward")
```

[View all reward models](https://modelscope.github.io/RM-Gallery/library/rm_library/)

### Custom Reward Models

Build your own reward models with simple APIs:

```python
from rm_gallery.core.reward import BasePointWiseReward

class CustomReward(BasePointWiseReward):
    def _evaluate(self, sample, **kwargs):
        # Your evaluation logic
        return RewardResult(...)
```

[Learn more about building custom RMs](https://modelscope.github.io/RM-Gallery/tutorial/building_rm/custom_reward/)

### Benchmarking

Evaluate models on standard benchmarks:

- **RewardBench2** - Latest reward model benchmark
- **RM-Bench** - Comprehensive evaluation suite
- **Conflict Detector** - Detect evaluation inconsistencies
- **JudgeBench** - Judge capability assessment

[Read the evaluation guide](https://modelscope.github.io/RM-Gallery/tutorial/evaluation/overview/)

### Applications

- **Best-of-N Selection** - Choose optimal responses from candidates
- **Data Refinement** - Improve dataset quality with reward signals
- **RLHF Integration** - Use rewards in reinforcement learning pipelines
- **High-Performance Serving** - Deploy models with fault-tolerant infrastructure

## Documentation

- [Quickstart Guide](https://modelscope.github.io/RM-Gallery/quickstart/)
- [Interactive Examples](./examples/)
- [Building Custom RMs](https://modelscope.github.io/RM-Gallery/tutorial/building_rm/custom_reward/)
- [Training Guide](https://modelscope.github.io/RM-Gallery/tutorial/training_rm/overview/)
- [API Reference](https://modelscope.github.io/RM-Gallery/api_reference/)

## Contributing

We welcome contributions! Please install pre-commit hooks before submitting pull requests:

```bash
pip install -e .
pre-commit install
```

See our [contribution guide](./docs/contribution.md) for details.

## Citation

If you use RM-Gallery in your research, please cite:

```
@software{
title = {RM-Gallery: A One-Stop Reward Model Platform},
author = {The RM-Gallery Team},
url = {https://github.com/modelscope/RM-Gallery},
month = {07},
year = {2025}
}
```
