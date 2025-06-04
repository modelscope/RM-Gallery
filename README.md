# RM-Gallery: A One-Stop Reward Model Platform

RM-Gallery is a one-stop platform for training, building and deploying reward models. It provides a comprehensive solution for implementing reward models at both task-level and component-level, with high-throughput and fault-tolerant capabilities.

## Features

### Training RM 
- **Integrated RM Training Pipeline**: Provides an RL-based framework for training reasoning reward models, compatible with popular frameworks (e.g., verl, OpenRLHF), and offers examples for integrating RM-Gallery into the framework. 

### Building RM 
- **Unified Reward Model Architecture**: Flexible implementation of reward models through standardized interfaces, supporting various architectures (model-based/free), reward formats (scalar/critique), and scoring patterns (pointwise/listwise/pairwise)

- **Comprehensive RM Gallery**: Provides a rich collection of ready-to-use Reward Model instances for diverse tasks (e.g., math, coding, preference alignment, agent) with both task-level(RMComposition) and component-level(RewardModel). Users can directly apply RMComposition for specific tasks or assemble custom RMComposition via component-level RewardModel. 

- **Principle-Critic-Score Paradigm**: Adopts the Principle+Critic+Score-based reasoning Reward Model  paradigm, offering best practices to help users generate principles with limited preference data.  

### Deploying RM 

- **Multiple Usage Scenarios**: Covers multiple Reward Model (RM) usage scenarios with detailed best practices, including Training with Rewards (e.g., post-training), Inference with Rewards (e.g., Best-of-N), and Post-Inference with Rewards (e.g., self-correction).

- **High-Performance RM Serving**: Leverages the New API platform to deliver high-throughput, fault-tolerant reward model serving, enhancing feedback efficiency. 


## 目录结构

```
Easy-RM/
├── easy_rm/                       # 主代码包
│   ├── __init__.py
│   ├── easy_rm.py                 # 主入口文件
│   ├── configs/                   # 配置文件
│   ├── data/                      # 数据处理
│   ├── galleries/                 # 画廊展示
│   ├── models/                    # 模型定义
│   │   ├── __init__.py
│   │   └── base_model.py
│   ├── prompt/                    # 提示模板
│   ├── scorers/                   # 评分器
│   ├── training/                  # 训练相关
│   ├── utils/                     # 工具函数
│   └── workers/                   # 工作进程
├── dataset/                       # 数据集
├── docs/                          # 文档
├── docker/                        # Docker配置
├── evaluation/                    # 评估工具
├── examples/                      # 示例代码
│   ├── __init__.py
│   ├── math/                      # 数学领域示例
│   ├── qa/                        # 问答领域示例
│   └── writing/                   # 写作领域示例
├── tests/                         # 测试代码
├── README.md                      # 项目说明
└── requirements.txt               # 依赖项
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用示例

（待补充）

## 贡献指南

（待补充）

## 许可证

（待补充）
