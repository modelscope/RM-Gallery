# RM-Gallery v2 框架

## 概述

v2框架是一个为AI模型评估而设计的下一代评估系统，具有灵活性和可扩展性。它提供了一种模块化的方法来定义、执行和分析各种评估任务。

## 核心组件

### 1. 模板定义 ([template.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/v2/model/template.py))

定义对话和模板结构的核心数据模型：

- `ChatMessage`：表示带有角色、内容和可选推理内容的单条消息
- `ChatTemplate`：定义带有占位符的消息模板，可以在运行时动态填充
- 支持多语言模板定义

### 2. 评估器 ([grader.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/v2/grader.py))

支持多种评估类型的灵活评估系统：

- `Grader`：所有评估函数的基类，支持Pointwise和Listwise评估模式
- `LLMGrader`：使用聊天模板的基于LLM的评估函数
- `FunctionReward`：基于自定义函数的评估实现
- `FactualGrader`：内置的事实准确性检查评估示例

### 3. 策略 ([strategy/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/v2/strategy/))

用于优化评估器性能的组件：

- `GraderOptimizer`：评估器优化器的基类
- `RepeatOptimizer`：通过重复执行并平均结果来优化评估器输出

### 4. 实验框架 ([experiment.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/v2/experiment.py))

用于进行评估实验的系统：

- `EvaluationExperiment`：编排数据集和评估器的评估过程
- 支持同步和异步评估
- 内置日志记录和结果跟踪

### 5. 数据集管理 ([dataset.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/v2/dataset.py))

处理具有模式验证的评估数据集：

- `EvaluationDataset`：管理评估样本集合
- 使用JSON Schema进行数据完整性验证
- 支持数据映射和转换

### 6. 评估器注册表 ([registry.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/v2/registry.py))

统一管理评估器的注册和获取：

- 支持命名空间分组管理
- 支持装饰器和直接注册两种方式
- 提供评估器的查询、列举和删除功能

## 功能特点

### 灵活的模板系统
定义可重用的对话模板，其中包含可在运行时填充的占位符变量。支持多语言模板和自动参数提取。

### 多种评估类型
支持各种评估函数，包括LLM-as-a-judge、基于评分标准的评估和自定义程序化评估。

### 评估器优化
提供优化器组件，可以提高评估器的稳定性和准确性，例如通过重复执行并平均结果。

### 异步处理
内置异步评估支持，以在处理LLM时最大化吞吐量。所有评估操作都支持异步执行。

### 模式验证
使用JSON Schema自动验证输入数据以确保数据质量，防止无效数据进入评估流程。

### 模块化设计
可组合的组件，可以针对不同的评估场景进行混合和匹配。各组件之间松耦合，易于扩展。

### 评估模式支持
支持Pointwise和Listwise两种评估模式：
- **Pointwise模式**：对每个样本进行独立评分，适用于直接质量评估场景
- **Listwise模式**：将所有样本一次性送入评估模型进行整体排名或评分，适用于全局排序任务

## 使用示例

### 基本使用

```
from rm_gallery.core.grader import FactualGrader
from rm_gallery.core.dataset import DataSample

# 创建一个简单的事实评估器
grader = FactualGrader()

# 准备数据
data_sample = DataSample(
    data={"query": "法国的首都是什么？"},
    samples=[{"answer": "巴黎"}, {"answer": "伦敦"}]
)

# 执行评估
results = await grader(data_sample)
```

### 使用策略

```
from rm_gallery.core.grader import FactualGrader
from rm_gallery.core.strategy.repeat import RepeatOptimizer
from rm_gallery.core.dataset import DataSample

# 创建一个评估器
grader = FactualGrader()

# 使用优化器包装评估器，重复执行5次并平均结果
optimized_grader = RepeatOptimizer(grader, num_repeats=5)

# 准备数据
data_sample = DataSample(
    data={"query": "法国的首都是什么？"},
    samples=[{"answer": "巴黎"}, {"answer": "伦敦"}]
)

# 执行优化后的评估
results = await optimized_grader(data_sample)
```

### 运行完整实验

```
from rm_gallery.core.dataset import EvaluationDataset
from rm_gallery.core.experiment import EvaluationExperiment
from rm_gallery.core.grader import FactualGrader

# 创建数据集
dataset = EvaluationDataset(
    data_sample_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "answer": {"type": "string"}
        },
        "required": ["query", "answer"]
    },
    data_samples=[
        {
            "data": {"query": "法国的首都是什么？"},
            "samples": [{"answer": "巴黎"}, {"answer": "马赛"}]
        },
        {
            "data": {"query": "德国的首都是什么？"},
            "samples": [{"answer": "柏林"}, {"answer": "慕尼黑"}]
        }
    ]
)

# 创建实验
experiment = EvaluationExperiment(graders=[FactualGrader()])

# 运行实验
result = await experiment(dataset)
```

### 自定义评估函数

```
from rm_gallery.core.grader import FunctionGrader, GraderScore
from rm_gallery.core.registry import GraderRegistry

# 定义自定义评估函数
async def custom_grader_function(**kwargs) -> GraderScore:
    # 自定义逻辑
    score = len(kwargs.get("answer", "")) / 100  # 示例逻辑
    return GraderScore(score=score, reason="基于答案长度的评分")

# 创建基于函数的评估器
custom_grader = FunctionGrader(
    name="length_based_grader",
    func=custom_grader_function,
    evaluation_mode="pointwise"
)

# 注册评估函数
GraderRegistry.register("length_grader", custom_grader, namespace="custom")

# 获取并使用评估函数
grader = GraderRegistry.get("custom.length_grader")
```

### 使用LLM评估函数

```
from rm_gallery.core.grader import LLMGrader
from rm_gallery.core.model.template import ChatTemplate
from rm_gallery.core.model.message import ChatMessage

# 定义评估模板
chat_template = ChatTemplate(
    messages=[
        ChatMessage(
            role="system",
            content="你是一个 helpful assistant，负责评估回答的质量。"
        ),
        ChatMessage(
            role="user",
            content="问题: {query}\n回答: {answer}\n请评估这个回答的质量，给出0-1之间的分数。"
        )
    ],
    model={
        "model_name": "gpt-3.5-turbo",
        "api_key": "your-api-key"
    }
)

# 创建LLM评估函数
llm_grader = LLMGrader(
    name="gpt_grader",
    chat=chat_template,
    evaluation_mode="pointwise"
)
```

## 快速开始

1. 在环境中配置您的LLM API凭证（通过环境变量或在[template.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/v2/model/template.py)中直接配置）
2. 使用适当的模式定义您的评估数据集
3. 使用内置类创建评估函数或使用自定义逻辑扩展
4. 运行实验以评估模型性能

## API参考

### Grader（评估器基类）

所有评估器的基类，定义了评估的基本接口和模式。

#### 属性
- `name` (str): 评估函数的名称
- `evaluation_mode` (GraderMode): 评估模式（POINTWISE 或 LISTWISE）

#### 方法
- `evaluate(**kwargs)`: 执行评估的核心方法，需要子类实现
- `__call__(data_sample)`: 调用评估器，处理数据样本

### GraderOptimizer（评估器优化器基类）

评估器优化器的基类，用于优化评估器的输出。

#### 属性
- `grader` (Grader | Callable): 被优化的评估器

#### 方法
- `__call__(data_sample)`: 执行优化逻辑

### ChatTemplate（聊天模板）

定义与LLM交互的模板。

#### 属性
- `messages` (List[ChatMessage]): 聊天消息列表
- `required` (List[str]): 必需的参数列表
- `model` (Dict): 模型配置参数

#### 方法
- `format(**kwargs)`: 格式化模板消息
- `__call__(model_output, **kwargs)`: 执行与LLM的交互

### GraderRegistry（评估器注册表）

管理所有评估函数的注册和获取。

#### 方法
- `register(name, grader, namespace)`: 注册评估函数
- `get(name)`: 获取评估函数
- `remove(name)`: 删除评估函数
- `list_graders(namespace)`: 列出评估函数
- `list_namespaces()`: 列出所有命名空间

## 未来开发

v2框架正在积极开发中，计划扩展：
- 更多内置评估类型
- 更多优化器实现
- 增强的分析和报告功能
- 与更多LLM提供商集成
- 高级实验跟踪功能
- 更完善的文档和示例