# Easy-RM: 大语言模型奖励模型框架

Easy-RM 是一个灵活且可扩展的大语言模型奖励模型框架，支持用户自定义评测原则，并提供高并发的LLM API调用能力。

## 特性

- 支持用户自定义评测原则
- 支持高并发LLM API调用
- 支持奖励模型训练
- 可扩展的架构设计

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
