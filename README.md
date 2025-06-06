# RM-Gallery: A One-Stop Reward Model Platform

RM-Gallery is a one-stop platform for training, building and deploying reward models. It provides a comprehensive solution for implementing reward models at both task-level and component-level, with high-throughput and fault-tolerant capabilities.

## 📢 News

## 🤔 Why RM-Gallery?

<p align="center">
 <img src="./docs/images/framework.png" alt="Framework" width="75%">
</p>

framework being optimized

### Training RM
- **Integrated RM Training Pipeline**: Provides an RL-based framework for training reasoning reward models, compatible with popular frameworks (e.g., verl, OpenRLHF), and offers examples for integrating RM-Gallery into the framework.

### Building RM
- **Unified Reward Model Architecture**: Flexible implementation of reward models through standardized interfaces, supporting various architectures (model-based/free), reward formats (scalar/critique), and scoring patterns (pointwise/listwise/pairwise)

- **Comprehensive RM Gallery**: Provides a rich collection of ready-to-use Reward Model instances for diverse tasks (e.g., math, coding, preference alignment, agent) with both task-level(RMComposition) and component-level(RewardModel). Users can directly apply RMComposition for specific tasks or assemble custom RMComposition via component-level RewardModel.

- **Principle-Critic-Score Paradigm**: Adopts the Principle+Critic+Score-based reasoning Reward Model  paradigm, offering best practices to help users generate principles with limited preference data.

### Deploying RM

- **Multiple Usage Scenarios**: Covers multiple Reward Model (RM) usage scenarios with detailed best practices, including Training with Rewards (e.g., post-training), Inference with Rewards (e.g., Best-of-N), and Post-Inference with Rewards (e.g., self-correction).

- **High-Performance RM Serving**: Leverages the New API platform to deliver high-throughput, fault-tolerant reward model serving, enhancing feedback efficiency.

## 📥 Installation
For installation, please refer to [Installation.md](docs/installation.md).

## 📚 Documentation
- Tutorial:
    - [installation](docs/tutorial/installation.md)
    - [quickstart](docs/tutorial/quickstart.md)
    - [rm_server](docs/tutorial/rm_server.md)
- Ready-to-use RMs & RMCompositions:
    - [RM Zoo](docs/RMs.md)
    - [RMComposition Zoo](docs/RMCompositions.md)
- RM Scenario Cases:
    - [RM Scenario Cases](docs/rm_scenario_cases.md)
- useful docs:
    - [RM-Gallery's Rankings in Popular Benchmark](docs/leaderboard.md)


## 🤝 Contribute

Contributions are always encouraged!

We highly recommend install pre-commit hooks in this repo before committing pull requests.
These hooks are small house-keeping scripts executed every time you make a git commit,
which will take care of the formatting and linting automatically.
```shell
pip install -e .
pre-commit install
```

Please refer to our [Contribution Guide](./docs/contribution.md) for more details.

## 📝 Citation

Reference to cite if you use RM-Gallery in a paper:

```
@software{
author = {GM-Gallery},
month = {06},
title = {GM-Gallery},
url = {},
year = {2025}
}
```