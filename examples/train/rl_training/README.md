# RL Training with RM-Gallery LLM Judge

使用 RM-Gallery 的 LLM as Judge 评估能力进行强化学习训练。

## 🎯 特性

- ✅ **6种评估模式**: Pairwise (winrate/copeland/dgr/elo), Pointwise, Listwise
- ✅ **框架通用**: 支持VERL, OpenRLHF等RL框架
- ✅ **灵活配置**: 通过环境变量轻松切换评估模式
- ✅ **自定义Prompt**: 完全支持自定义评估提示词
- ✅ **并发评估**: 高效的并行LLM调用

## 📁 文件结构

```
rl_training/
├── README.md                    # 本文档
├── alignment_reward_fn.py      # RM-Gallery奖励函数
├── alignment_rl_dataset.py     # 数据集类
├── reward_manager.py           # DGR Reward Manager
├── grpo_training.sh            # GRPO训练脚本
├── config_example.yaml         # 配置示例
└── data/                       # 训练数据
    ├── wildchat_10k_train.parquet
    └── wildchat_10k_test.parquet
```

## 🚀 快速开始

### 步骤 1: 配置Judge API

编辑 `grpo_training.sh` 中的Judge配置：

```bash
# Judge Model Configuration
export JUDGE_MODEL_NAME="qwen3-32b"
export JUDGE_API_URL="http://your-api-url/v1/chat/completions"
export JUDGE_API_KEY="your-api-key"

# Evaluation Mode
export EVAL_MODE="pairwise"        # pairwise, pointwise, listwise
export PAIRWISE_MODE="dgr"         # dgr, copeland, winrate, elo
```

### 步骤 2: 注册Reward Manager (首次使用)

将 `reward_manager.py` 复制到VERL：

```bash
cp reward_manager.py $VERL_ROOT/verl/workers/reward_manager/dgr.py
```

编辑 `$VERL_ROOT/verl/workers/reward_manager/__init__.py`，添加：

```python
from .dgr import DGRRewardManager
```

只需配置一次！

### 步骤 3: 配置训练参数

编辑 `grpo_training.sh` 中的训练配置：

```bash
# VERL Root Directory
VERL_ROOT="/path/to/verl"

# Model Paths
ACTOR_MODEL_PATH="/path/to/base/model"

# Data Paths
TRAIN_DATA="examples/train/rl_training/data/wildchat_10k_train.parquet"
VAL_DATA="examples/train/rl_training/data/wildchat_10k_test.parquet"
```

### 步骤 4: 启动训练

```bash
chmod +x grpo_training.sh
./grpo_training.sh
```

## 🔧 评估模式详解

### 1. Pairwise - DGR 模式（推荐）

使用TFAS算法解决循环冲突，最准确的评估方式。

```bash
export EVAL_MODE="pairwise"
export PAIRWISE_MODE="dgr"
```

**特点**：
- 两两比较所有响应
- 检测并解决循环冲突（如A>B>C但A<C）
- 计算基于无冲突图的净胜数
- 适合：追求高质量评估的正式训练

### 2. Pairwise - Winrate 模式

简单的胜率统计，速度快。

```bash
export EVAL_MODE="pairwise"
export PAIRWISE_MODE="winrate"
```

**特点**：
- 两两比较所有响应
- 统计胜率：wins / total_comparisons
- 不处理循环冲突
- 适合：快速实验和调试

### 3. Pointwise 模式

独立打分（1-10分），速度最快。

```bash
export EVAL_MODE="pointwise"
```

**特点**：
- 对每个响应独立打分
- 无需两两比较
- 速度快，适合大规模训练
- 适合：快速迭代实验

### 4. Listwise 模式

一次性对所有响应排序。

```bash
export EVAL_MODE="listwise"
```

**特点**：
- 全局排序，考虑所有响应
- 适合3-6个响应的场景
- 适合：多候选排序任务

## 📝 数据格式

数据格式与原始DGR完全兼容，使用Parquet格式：

```python
{
    "x": [{"role": "user", "content": "用户问题"}],
    "chosen": [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": "更好的回复"}
    ],
    "rejected": [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": "较差的回复"}
    ],
    "source": "data_source",
    "unique_id": "unique_identifier"
}
```

**无需修改数据**，只需替换reward计算即可！

## 🎨 自定义Prompt

编辑 `alignment_reward_fn.py` 中的 `custom_alignment_prompt` 函数：

```python
def custom_alignment_prompt(user_query, response_a, response_b, reference="", **kwargs):
    """自定义评估提示词"""
    return f"""评估两个回复的质量...

问题：{user_query}
回复A：{response_a}
回复B：{response_b}

<result>A/B/tie</result>
"""
```

## ⚙️ 高级配置

### 性能优化

```bash
# 增加并发数（根据API限制调整）
export MAX_WORKERS=20

# 调整生成数量
N_SAMPLES_PER_PROMPT=6
```

### 调试模式

```bash
# 启用详细日志
export VERBOSE="true"
```

### 使用不同算法

GRPO训练：
```bash
./grpo_training.sh
```

GSPO训练：
```bash
# 修改 grpo_training.sh，或创建 gspo_training.sh
algorithm.name="gspo"
```

## 📊 训练监控

训练过程中会输出评估信息：

```
📊 使用Pairwise-DGR评估模式
📊 Chosen评分: 0.72
📊 模型平均评分: 0.69
📊 超越Chosen的比例: 35.4%
📊 冲突解决数: 5
```

## 🔍 DGR算法说明

DGR = TFAS (Tournament Feedback Arc Set)

**算法流程**：
1. 构建Tournament图（有向图）
2. 检测循环冲突
3. 移除最小边集消除所有循环
4. 计算净胜数
5. 归一化到[-1, 1]

**算法选择**：
- n ≤ 10: 精确算法（枚举所有排列）
- n > 10: 贪心算法（基于初始净胜数）

## 📈 与原DGR的区别

| 项目 | 原DGR | RM-Gallery版本 |
|------|-------|---------------|
| 核心算法 | ✓ TFAS | ✓ TFAS（完整迁移） |
| 评估模式 | Pairwise | Pairwise/Pointwise/Listwise |
| LLM调用 | 直接API | RM-Gallery统一接口 |
| 自定义 | 部分 | 完全支持 |
| 代码结构 | 单文件 | 模块化 |
| 数据格式 | 固定 | **完全兼容** |

**关键优势**：
- ✅ **零数据迁移**：数据格式完全兼容
- ✅ **更多模式**：支持pointwise/listwise
- ✅ **易于扩展**：模块化设计
- ✅ **统一接口**：使用RM-Gallery生态

## 🐛 故障排查

### 问题1：API调用失败

检查API配置：
```bash
export JUDGE_API_URL="http://correct-url/v1/chat/completions"
export JUDGE_API_KEY="correct-api-key"
```

### 问题2：评估速度慢

增加并发数：
```bash
export MAX_WORKERS=20
```

减少生成数：
```bash
N_SAMPLES_PER_PROMPT=2
```

### 问题3：显存不足

减小batch size：
```bash
TRAIN_BATCH_SIZE=32
```

### 问题4：找不到reward函数

检查路径：
```bash
REWARD_FN_PATH="examples/train/rl_training/alignment_reward_fn.py"
```

## 📚 相关文档

- [RM-Gallery VERL Integration](../../integrations/verl/README.md)
- [VERL Documentation](https://github.com/volcengine/verl)
- [DGR Algorithm Paper](链接到相关论文)

## 🤝 贡献

欢迎提交Issue和PR！

## 📄 License

Follow RM-Gallery's license.

