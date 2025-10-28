# RM-Gallery vs 主流 Eval 框架对比分析报告

**文档版本**: v1.0
**创建日期**: 2025-10-24
**分析对象**: OpenAI Evals, LangChain OpenEvals, Google ADK, Azure AI Agent Evals
**分析维度**: 功能完整性、架构设计、工程化能力

---

## 📑 目录

1. [执行摘要](#执行摘要)
2. [分析方法](#分析方法)
3. [框架概览](#框架概览)
4. [关键能力对比](#关键能力对比)
5. [缺失能力详细分析](#缺失能力详细分析)
6. [优先级建议](#优先级建议)
7. [实施路线图](#实施路线图)
8. [差异化定位建议](#差异化定位建议)

---

## 执行摘要

### 核心发现

通过深度分析四个主流 LLM 评估框架的源代码和架构设计，我们发现了**10个关键能力领域**，其中 RM-Gallery 在以下方面存在显著缺失：

1. **用户模拟器 (User Simulator)** - 所有对比框架都具备 ⭐⭐⭐
2. **Eval Case/Set 管理系统** - 所有对比框架都具备 ⭐⭐⭐
3. **统计分析和对比** - Azure AI Evals 的核心优势 ⭐⭐⭐
4. **Trajectory/Tool 评估** - Google ADK 的核心能力 ⭐⭐⭐
5. **代码评估能力** - OpenEvals 的独特优势 ⭐⭐

### 战略建议

RM-Gallery 应该在保持其核心优势（Reward Model 专业性、Rubric 系统、Training 集成）的基础上，补齐通用评估框架的基础能力，将自身定位为：

> **"专业的 Reward Model 平台 + 完整的 LLM 评估框架"**

### 优先级矩阵

| 能力 | 紧急度 | 重要度 | 建议优先级 |
|------|--------|--------|------------|
| Eval Case/Set 管理系统 | 高 | 高 | P0 |
| 用户模拟器 | 高 | 高 | P0 |
| 统计分析和对比 | 高 | 高 | P0 |
| Trajectory/Tool 评估 | 中 | 高 | P1 |
| RAG 专用评估 | 中 | 中 | P1 |

---

## 分析方法

### 研究对象

我们选择了四个具有代表性的评估框架进行深度对比：

1. **OpenAI Evals** - OpenAI 官方评估框架，广泛应用于 GPT 系列模型评估
2. **LangChain OpenEvals** - LangChain 生态的评估工具，专注于应用场景
3. **Google ADK (Agent Development Kit)** - Google 的 Agent 开发和评估框架
4. **Azure AI Agent Evals** - Microsoft Azure 的 AI Agent 评估解决方案

### 分析维度

- **架构设计**: 核心抽象、扩展性、模块化程度
- **功能完整性**: 评估类型、使用场景覆盖度
- **工程化能力**: CI/CD 集成、可维护性、团队协作
- **生态系统**: 与其他工具的集成能力

### 数据来源

- 框架源代码深度分析 (位于 `data/eval_examples/`)
- 官方文档和 README
- 架构设计和接口定义

---

## 框架概览

### OpenAI Evals

**定位**: 通用 LLM 评估框架，支持模型能力基准测试

**核心特点**:
- 强大的 Registry 系统 (evals, data, eval_sets, completion_fns)
- Solver 抽象，支持复杂推理链评估
- YAML 驱动的配置管理
- 丰富的内置评估套件 (elsuite)

**架构亮点**:
```python
class Eval(abc.ABC):
    def __init__(self, completion_fns: list[Union[CompletionFn, Solver]]):
        # 支持多种 completion function
        self.completion_fns = [maybe_wrap_with_compl_fn(fn) for fn in completion_fns]

    @abc.abstractmethod
    def eval_sample(self, sample: Any, rng: random.Random):
        """评估单个样本"""

    @abc.abstractmethod
    def run(self, recorder: RecorderBase) -> Dict[str, float]:
        """运行完整评估"""
```

**典型应用场景**:
- 模型基准测试 (MMLU, HumanEval, etc.)
- 推理能力评估 (CoT, ReAct)
- 自定义评估任务

---

### LangChain OpenEvals

**定位**: 应用导向的评估工具，专注于生产环境 LLM 应用

**核心特点**:
- LLM-as-Judge 核心范式
- 丰富的预构建评估器 (Correctness, Conciseness, Hallucination, etc.)
- RAG 专用评估器套件
- 代码评估能力 (Pyright, Mypy, Sandboxed Execution)
- 多轮对话模拟 (Multiturn Simulation)

**架构亮点**:
```python
def create_llm_as_judge(
    *,
    prompt: Union[str, Runnable, Callable],
    judge: Optional[Union[ModelClient, BaseChatModel]] = None,
    continuous: bool = False,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleEvaluator:
    """创建 LLM-as-Judge 评估器"""
```

**典型应用场景**:
- RAG 应用质量评估
- 代码生成任务评估
- 多轮对话系统评估
- 结构化输出验证

---

### Google ADK

**定位**: Agent 开发和评估的完整工具包

**核心特点**:
- 完整的 Agent 评估生命周期管理
- 用户模拟器 (User Simulator)
- Trajectory 评估 (Agent 执行轨迹)
- Tool Use Quality 评估
- 多种存储后端 (Local, GCS, Memory, Vertex AI)
- Rubric-based 评估

**架构亮点**:
```python
# Evaluator 接口
class Evaluator(ABC):
    def evaluate_invocations(
        self,
        actual_invocations: list[Invocation],
        expected_invocations: list[Invocation],
    ) -> EvaluationResult:
        """评估 Agent 调用序列"""

# User Simulator 接口
class UserSimulator(ABC):
    async def get_next_user_message(
        self, events: list[Event]
    ) -> NextUserMessage:
        """生成下一条用户消息"""
```

**典型应用场景**:
- Agent 能力评估
- 多轮交互自动化测试
- Tool calling 质量评估
- Agent 推理过程分析

---

### Azure AI Agent Evals

**定位**: 企业级 AI Agent 评估解决方案，深度集成 Azure 生态

**核心特点**:
- GitHub Action 开箱即用
- 统计分析能力 (置信区间、显著性检验)
- A/B 测试支持 (Baseline vs Treatment)
- 系统化 Safety 评估
- Azure AI Studio 集成

**架构亮点**:
```python
# 统计分析
class EvaluationScoreCI:
    def _compute_ci(self, data: pd.Series, confidence_level: float = 0.95):
        if self.score.data_type == EvaluationScoreDataType.BOOLEAN:
            result = binomtest(data.sum(), data.count())
            ci = result.proportion_ci(confidence_level, method="wilsoncc")
        elif self.score.data_type == EvaluationScoreDataType.CONTINUOUS:
            # t-distribution for continuous data
            stderr = data.std() / (self.count**0.5)
            z_ao2 = t.ppf(1 - (1 - confidence_level) / 2, df=self.count - 1)

# 统计显著性检验
def mcnemar(contingency_table: np.ndarray) -> float:
    """McNemar's test for paired boolean data"""
```

**典型应用场景**:
- CI/CD 集成的自动化评估
- 科学严谨的模型对比实验
- 企业级安全风险评估
- 团队协作和结果共享

---

## 关键能力对比

### 能力矩阵

| 能力维度 | RM-Gallery | OpenAI Evals | OpenEvals | Google ADK | Azure AI Evals | 所有框架共性 |
|---------|------------|--------------|-----------|------------|----------------|--------------|
| **基础评估能力** |
| Pointwise 评估 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pairwise 评估 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Listwise 评估 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| LLM-as-Judge | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **测试管理** |
| Eval Case 管理 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Eval Set 管理 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Registry 系统 | ⚠️ (仅 RM) | ✅ | ✅ | ✅ | ✅ | ✅ |
| 结果管理 | ⚠️ (简单) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **高级评估** |
| User Simulator | ❌ | ❌ | ✅ | ✅ | ❌ | ⚠️ |
| Trajectory 评估 | ❌ | ⚠️ | ❌ | ✅ | ❌ | ❌ |
| Tool Use 评估 | ❌ | ⚠️ | ❌ | ✅ | ✅ | ⚠️ |
| RAG 专用评估 | ❌ | ❌ | ✅ | ⚠️ | ❌ | ❌ |
| 代码评估 | ❌ | ⚠️ | ✅ | ❌ | ❌ | ❌ |
| **统计分析** |
| 置信区间 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| 显著性检验 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| A/B 测试 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **工程化** |
| CI/CD 集成 | ❌ | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ |
| 多存储后端 | ❌ | ⚠️ | ❌ | ✅ | ✅ | ⚠️ |
| 可视化报告 | ⚠️ (简单) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **RM-Gallery 独有** |
| Rubric 生成 | ✅ | ❌ | ❌ | ⚠️ | ❌ | - |
| RM 训练集成 | ✅ | ❌ | ❌ | ❌ | ❌ | - |
| 高性能 Serving | ✅ | ❌ | ❌ | ⚠️ | ❌ | - |

**图例**: ✅ 完整支持 | ⚠️ 部分支持 | ❌ 不支持

---

## 缺失能力详细分析

### 1. 用户模拟器 (User Simulator) ⭐⭐⭐

#### 问题描述

**所有对比框架都具备，RM-Gallery 完全缺失**

用户模拟器是自动化多轮对话评估的关键组件。当前 RM-Gallery 只能评估单次交互（一问一答），无法测试：
- 多轮对话的连贯性
- 上下文理解能力
- 长对话中的记忆保持
- 复杂任务的分解和执行

#### 对比框架实现

**Google ADK 的实现**:

```python
class UserSimulator(ABC):
    """用户模拟器基类"""

    async def get_next_user_message(
        self, events: list[Event]
    ) -> NextUserMessage:
        """根据历史交互生成下一条用户消息

        Args:
            events: Agent 的历史执行事件

        Returns:
            NextUserMessage: 包含状态和下一条消息
        """
        pass

class Status(enum.Enum):
    SUCCESS = "success"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    STOP_SIGNAL_DETECTED = "stop_signal_detected"
    NO_MESSAGE_GENERATED = "no_message_generated"
```

**LangChain OpenEvals 的实现**:

```python
from openevals.simulators import simulate_user

# 自动化多轮对话
async for turn in simulate_user(
    user_simulator=my_simulator,
    agent=my_agent,
    max_turns=10
):
    print(f"Turn {turn.turn_number}: {turn.message}")
```

#### 影响分析

**业务影响**:
- 🚫 无法评估多轮对话能力
- 🚫 无法测试长对话场景
- 🚫 无法自动化交互式任务测试
- 🚫 无法评估 Agent 的任务规划能力

**技术债务**:
- 只能依赖人工测试多轮对话
- 无法进行大规模自动化测试
- 评估结果不够全面

#### 实现建议

**Phase 1: 基础用户模拟器**

```python
from rm_gallery.core.evaluation import BaseUserSimulator, UserMessage

class StaticUserSimulator(BaseUserSimulator):
    """预定义脚本的用户模拟器"""

    def __init__(self, script: list[str]):
        self.script = script
        self.current_turn = 0

    async def get_next_message(
        self, history: list[Message]
    ) -> UserMessage | None:
        if self.current_turn >= len(self.script):
            return None
        message = self.script[self.current_turn]
        self.current_turn += 1
        return UserMessage(content=message)

class LLMUserSimulator(BaseUserSimulator):
    """LLM 驱动的动态用户模拟器"""

    def __init__(self, task_description: str, llm: BaseLLM):
        self.task_description = task_description
        self.llm = llm

    async def get_next_message(
        self, history: list[Message]
    ) -> UserMessage | None:
        # 根据任务描述和历史对话生成下一条消息
        prompt = self._build_prompt(history)
        response = await self.llm.generate(prompt)
        return self._parse_response(response)
```

**Phase 2: 多轮评估框架**

```python
class MultiturnEvaluator:
    """多轮对话评估器"""

    async def evaluate_multiturn(
        self,
        agent: BaseAgent,
        simulator: BaseUserSimulator,
        max_turns: int = 10,
        evaluators: list[BaseReward] = None
    ) -> MultiturnResult:
        """执行多轮对话评估"""
        history = []

        for turn in range(max_turns):
            # 用户模拟器生成消息
            user_msg = await simulator.get_next_message(history)
            if user_msg is None:
                break

            # Agent 响应
            agent_response = await agent.respond(user_msg, history)

            # 记录历史
            history.extend([user_msg, agent_response])

            # 评估每个回合
            turn_result = self._evaluate_turn(
                user_msg, agent_response, evaluators
            )

        return self._aggregate_results(history)
```

**预计工作量**: 2-3 周 (1 个工程师)

---

### 2. Eval Case/Set 管理系统 ⭐⭐⭐

#### 问题描述

**所有对比框架都具备完整的测试用例管理系统**

当前 RM-Gallery 的评估是针对特定 benchmark（如 RewardBench2, JudgeBench）硬编码的，缺乏：
- 通用的测试用例定义和管理
- 测试集合的组织和版本控制
- 结果的持久化和检索
- 测试用例的复用和共享

#### 对比框架实现

**Google ADK 的实现**:

```python
@dataclass
class EvalCase:
    """单个测试用例"""
    id: str
    inputs: dict  # 输入数据
    expected_outputs: dict  # 期望输出
    metadata: dict = None  # 元数据

class EvalSet:
    """测试集合"""
    id: str
    name: str
    description: str
    eval_cases: list[EvalCase]

class EvalSetsManager:
    """测试集管理器"""
    def save_eval_set(self, eval_set: EvalSet) -> None:
        """保存测试集"""

    def load_eval_set(self, eval_set_id: str) -> EvalSet:
        """加载测试集"""

    def list_eval_sets(self) -> list[str]:
        """列出所有测试集"""
```

**OpenAI Evals 的 Registry 系统**:

```yaml
# evals/registry/evals/my_eval.yaml
my_eval:
  id: my_eval.v1
  description: "测试数学能力"
  metrics: [accuracy]

my_eval.train:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: train_samples.jsonl

my_eval.test:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: test_samples.jsonl
```

#### 影响分析

**业务影响**:
- 🚫 难以系统化管理测试用例
- 🚫 无法构建可复用的评估套件
- 🚫 团队协作困难（无法共享测试用例）
- 🚫 无法追踪历史评估结果

**技术债务**:
- 每个 benchmark 都需要单独实现加载逻辑
- 测试用例分散，难以维护
- 无法进行版本控制

#### 实现建议

**Phase 1: 核心数据模型**

```python
from pydantic import BaseModel
from typing import Any, Optional
from datetime import datetime

class EvalCase(BaseModel):
    """测试用例"""
    id: str
    inputs: dict[str, Any]
    expected_outputs: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = {}
    tags: list[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "id": "math_001",
                "inputs": {
                    "query": "What is 2+2?",
                    "context": "Basic arithmetic"
                },
                "expected_outputs": {
                    "answer": "4"
                },
                "tags": ["math", "basic"]
            }
        }

class EvalSet(BaseModel):
    """测试集合"""
    id: str
    name: str
    description: str
    version: str = "1.0"
    eval_cases: list[EvalCase]
    created_at: datetime
    metadata: dict[str, Any] = {}

class EvalResult(BaseModel):
    """评估结果"""
    eval_set_id: str
    eval_case_id: str
    model_name: str
    reward_name: str
    score: float
    details: dict[str, Any]
    timestamp: datetime
```

**Phase 2: 管理器实现**

```python
from abc import ABC, abstractmethod
import json
from pathlib import Path

class BaseEvalSetsManager(ABC):
    """测试集管理器基类"""

    @abstractmethod
    def save_eval_set(self, eval_set: EvalSet) -> None:
        pass

    @abstractmethod
    def load_eval_set(self, eval_set_id: str) -> EvalSet:
        pass

    @abstractmethod
    def list_eval_sets(self) -> list[str]:
        pass

class LocalEvalSetsManager(BaseEvalSetsManager):
    """本地文件系统管理器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_eval_set(self, eval_set: EvalSet) -> None:
        path = self.base_dir / f"{eval_set.id}.json"
        with open(path, 'w') as f:
            json.dump(eval_set.model_dump(), f, indent=2, default=str)

    def load_eval_set(self, eval_set_id: str) -> EvalSet:
        path = self.base_dir / f"{eval_set_id}.json"
        with open(path, 'r') as f:
            data = json.load(f)
        return EvalSet(**data)

    def list_eval_sets(self) -> list[str]:
        return [p.stem for p in self.base_dir.glob("*.json")]

class EvalResultsManager:
    """评估结果管理器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: EvalResult) -> None:
        """保存单个评估结果"""
        path = self._get_result_path(result)
        with open(path, 'a') as f:
            f.write(result.model_dump_json() + '\n')

    def load_results(
        self,
        eval_set_id: str,
        model_name: Optional[str] = None
    ) -> list[EvalResult]:
        """加载评估结果"""
        results = []
        pattern = f"{eval_set_id}_{model_name or '*'}_*.jsonl"
        for path in self.base_dir.glob(pattern):
            with open(path, 'r') as f:
                for line in f:
                    results.append(EvalResult.model_validate_json(line))
        return results
```

**Phase 3: YAML 配置支持**

```python
class YAMLEvalSetLoader:
    """从 YAML 文件加载测试集"""

    @staticmethod
    def load_from_yaml(yaml_path: Path) -> EvalSet:
        """从 YAML 配置加载测试集

        Example YAML:
        ```yaml
        id: math_basic
        name: "Basic Math Evaluation"
        description: "Tests basic arithmetic"
        version: "1.0"

        cases:
          - id: case_001
            inputs:
              query: "What is 2+2?"
            expected_outputs:
              answer: "4"
            tags: [math, arithmetic]

          - id: case_002
            inputs:
              query: "Calculate 10 * 5"
            expected_outputs:
              answer: "50"
            tags: [math, multiplication]
        ```
        """
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        cases = [EvalCase(**case) for case in data.pop('cases', [])]
        return EvalSet(eval_cases=cases, **data)
```

**Phase 4: CLI 工具**

```bash
# 列出所有测试集
rm-gallery eval list

# 创建新测试集
rm-gallery eval create --name "my_test" --from-jsonl data.jsonl

# 运行评估
rm-gallery eval run --eval-set math_basic --reward math_rm

# 查看结果
rm-gallery eval results --eval-set math_basic --model gpt-4
```

**预计工作量**: 3-4 周 (1 个工程师)

---

### 3. 统计分析和对比 ⭐⭐⭐

#### 问题描述

**Azure AI Evals 的核心优势，RM-Gallery 完全缺失**

当前 RM-Gallery 只能计算简单的准确率，缺乏科学的统计分析：
- 无置信区间估计
- 无统计显著性检验
- 无法判断改进是否显著
- 无法进行严谨的 A/B 测试

这导致无法科学评估：
- 模型 A 是否真的比模型 B 好？
- 新版本的改进是否具有统计显著性？
- 需要多少测试样本才能得出可靠结论？

#### 对比框架实现

**Azure AI Evals 的统计分析**:

```python
from scipy.stats import binom, binomtest, t, ttest_rel, wilcoxon

class EvaluationScoreCI:
    """置信区间计算"""

    def _compute_ci(self, data: pd.Series, confidence_level: float = 0.95):
        """计算置信区间"""

        if self.score.data_type == EvaluationScoreDataType.BOOLEAN:
            # Boolean 数据: Wilson Score Interval
            result = binomtest(data.sum(), data.count())
            ci = result.proportion_ci(
                confidence_level=confidence_level,
                method="wilsoncc"
            )
            return ci.low, result.proportion_estimate, ci.high

        elif self.score.data_type == EvaluationScoreDataType.CONTINUOUS:
            # Continuous 数据: t-distribution
            mean = data.mean()
            stderr = data.std() / (self.count**0.5)
            z_ao2 = t.ppf(1 - (1 - confidence_level) / 2, df=self.count - 1)
            ci_lower = mean - z_ao2 * stderr
            ci_upper = mean + z_ao2 * stderr
            return ci_lower, mean, ci_upper

def mcnemar(contingency_table: np.ndarray) -> float:
    """McNemar's test for paired boolean data

    用于比较两个模型在相同测试集上的表现差异
    """
    n12 = contingency_table[0, 1]  # Model A 对，Model B 错
    n21 = contingency_table[1, 0]  # Model A 错，Model B 对
    n = n12 + n21

    # Mid-p version
    pvalue_exact = 2 * binom.cdf(k=min(n12, n21), n=n, p=0.5)
    pvalue_midp = pvalue_exact - binom.pmf(k=n12, n=n, p=0.5)

    return float(pvalue_midp)
```

#### 影响分析

**业务影响**:
- 🚫 无法科学评估模型改进效果
- 🚫 无法判断实验结果的可靠性
- 🚫 无法确定所需的测试样本量
- 🚫 难以向 stakeholder 提供有说服力的数据

**技术债务**:
- 评估结果可信度低
- 无法进行科学的对比实验
- 容易得出错误结论

#### 实现建议

**Phase 1: 置信区间计算**

```python
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.stats import binom, binomtest, t

class ScoreType(Enum):
    BOOLEAN = "boolean"      # True/False
    CONTINUOUS = "continuous"  # 0.0 - 1.0
    ORDINAL = "ordinal"       # 1, 2, 3, 4, 5

@dataclass
class ConfidenceInterval:
    """置信区间"""
    mean: float
    lower: float
    upper: float
    confidence_level: float
    sample_size: int
    score_type: ScoreType

class StatisticsAnalyzer:
    """统计分析器"""

    @staticmethod
    def compute_ci(
        scores: list[float],
        score_type: ScoreType,
        confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """计算置信区间"""

        scores = np.array(scores)
        n = len(scores)
        mean = scores.mean()

        if score_type == ScoreType.BOOLEAN:
            # Wilson Score Interval for binary data
            result = binomtest(int(scores.sum()), n)
            ci = result.proportion_ci(
                confidence_level=confidence_level,
                method="wilsoncc"
            )
            return ConfidenceInterval(
                mean=result.proportion_estimate,
                lower=ci.low,
                upper=ci.high,
                confidence_level=confidence_level,
                sample_size=n,
                score_type=score_type
            )

        elif score_type == ScoreType.CONTINUOUS:
            # t-distribution for continuous data
            stderr = scores.std() / np.sqrt(n)
            t_critical = t.ppf(1 - (1 - confidence_level) / 2, df=n - 1)
            margin = t_critical * stderr

            return ConfidenceInterval(
                mean=mean,
                lower=mean - margin,
                upper=mean + margin,
                confidence_level=confidence_level,
                sample_size=n,
                score_type=score_type
            )

        elif score_type == ScoreType.ORDINAL:
            # For ordinal data, CI is not meaningful
            return ConfidenceInterval(
                mean=mean,
                lower=None,
                upper=None,
                confidence_level=confidence_level,
                sample_size=n,
                score_type=score_type
            )
```

**Phase 2: 显著性检验**

```python
from scipy.stats import ttest_rel, wilcoxon, chi2_contingency

@dataclass
class SignificanceTestResult:
    """显著性检验结果"""
    test_name: str
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: Optional[float] = None

class SignificanceTest:
    """显著性检验"""

    @staticmethod
    def mcnemar_test(
        model_a_results: list[bool],
        model_b_results: list[bool],
        alpha: float = 0.05
    ) -> SignificanceTestResult:
        """McNemar's test for paired binary data

        用于比较两个模型在相同测试集上的差异
        """
        assert len(model_a_results) == len(model_b_results)

        # 构建 2x2 contingency table
        #              Model B Correct  Model B Wrong
        # Model A Correct    n11             n12
        # Model A Wrong      n21             n22

        n11 = sum(a and b for a, b in zip(model_a_results, model_b_results))
        n12 = sum(a and not b for a, b in zip(model_a_results, model_b_results))
        n21 = sum(not a and b for a, b in zip(model_a_results, model_b_results))
        n22 = sum(not a and not b for a, b in zip(model_a_results, model_b_results))

        # McNemar's test focuses on discordant pairs
        n = n12 + n21
        if n == 0:
            return SignificanceTestResult(
                test_name="McNemar",
                p_value=1.0,
                is_significant=False,
                alpha=alpha
            )

        # Mid-p version
        pvalue_exact = 2 * binom.cdf(k=min(n12, n21), n=n, p=0.5)
        pvalue_midp = pvalue_exact - binom.pmf(k=n12, n=n, p=0.5)

        return SignificanceTestResult(
            test_name="McNemar",
            p_value=pvalue_midp,
            is_significant=pvalue_midp < alpha,
            alpha=alpha
        )

    @staticmethod
    def paired_t_test(
        model_a_scores: list[float],
        model_b_scores: list[float],
        alpha: float = 0.05
    ) -> SignificanceTestResult:
        """Paired t-test for continuous scores"""
        assert len(model_a_scores) == len(model_b_scores)

        statistic, p_value = ttest_rel(model_a_scores, model_b_scores)

        # Cohen's d for effect size
        diff = np.array(model_a_scores) - np.array(model_b_scores)
        effect_size = diff.mean() / diff.std()

        return SignificanceTestResult(
            test_name="Paired t-test",
            p_value=p_value,
            is_significant=p_value < alpha,
            alpha=alpha,
            effect_size=abs(effect_size)
        )

    @staticmethod
    def wilcoxon_test(
        model_a_scores: list[float],
        model_b_scores: list[float],
        alpha: float = 0.05
    ) -> SignificanceTestResult:
        """Wilcoxon signed-rank test (non-parametric)"""
        assert len(model_a_scores) == len(model_b_scores)

        statistic, p_value = wilcoxon(model_a_scores, model_b_scores)

        return SignificanceTestResult(
            test_name="Wilcoxon",
            p_value=p_value,
            is_significant=p_value < alpha,
            alpha=alpha
        )
```

**Phase 3: A/B 测试框架**

```python
@dataclass
class ABTestResult:
    """A/B 测试结果"""
    baseline_name: str
    treatment_name: str
    baseline_ci: ConfidenceInterval
    treatment_ci: ConfidenceInterval
    significance_test: SignificanceTestResult
    improvement: float
    recommendation: str

class ABTester:
    """A/B 测试器"""

    def compare_models(
        self,
        baseline_scores: list[float],
        treatment_scores: list[float],
        score_type: ScoreType,
        baseline_name: str = "Baseline",
        treatment_name: str = "Treatment",
        alpha: float = 0.05
    ) -> ABTestResult:
        """对比两个模型"""

        # 计算置信区间
        baseline_ci = StatisticsAnalyzer.compute_ci(
            baseline_scores, score_type
        )
        treatment_ci = StatisticsAnalyzer.compute_ci(
            treatment_scores, score_type
        )

        # 选择合适的显著性检验
        if score_type == ScoreType.BOOLEAN:
            baseline_bool = [s > 0.5 for s in baseline_scores]
            treatment_bool = [s > 0.5 for s in treatment_scores]
            sig_test = SignificanceTest.mcnemar_test(
                baseline_bool, treatment_bool, alpha
            )
        elif score_type == ScoreType.CONTINUOUS:
            sig_test = SignificanceTest.paired_t_test(
                baseline_scores, treatment_scores, alpha
            )
        else:
            sig_test = SignificanceTest.wilcoxon_test(
                baseline_scores, treatment_scores, alpha
            )

        # 计算改进幅度
        improvement = (treatment_ci.mean - baseline_ci.mean) / baseline_ci.mean

        # 生成建议
        if sig_test.is_significant:
            if improvement > 0:
                recommendation = f"✅ Treatment 显著优于 Baseline (p={sig_test.p_value:.4f})"
            else:
                recommendation = f"⚠️ Treatment 显著差于 Baseline (p={sig_test.p_value:.4f})"
        else:
            recommendation = f"➖ 两者无显著差异 (p={sig_test.p_value:.4f})"

        return ABTestResult(
            baseline_name=baseline_name,
            treatment_name=treatment_name,
            baseline_ci=baseline_ci,
            treatment_ci=treatment_ci,
            significance_test=sig_test,
            improvement=improvement,
            recommendation=recommendation
        )
```

**Phase 4: 可视化报告**

```python
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticsVisualizer:
    """统计可视化"""

    @staticmethod
    def plot_ab_test(result: ABTestResult, save_path: Path = None):
        """绘制 A/B 测试结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 置信区间对比
        models = [result.baseline_name, result.treatment_name]
        means = [result.baseline_ci.mean, result.treatment_ci.mean]
        lowers = [result.baseline_ci.lower, result.treatment_ci.lower]
        uppers = [result.baseline_ci.upper, result.treatment_ci.upper]

        ax1.errorbar(
            models, means,
            yerr=[
                [m - l for m, l in zip(means, lowers)],
                [u - m for m, u in zip(means, uppers)]
            ],
            fmt='o', markersize=10, capsize=5
        )
        ax1.set_title('Model Comparison with 95% CI')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)

        # 显著性标注
        if result.significance_test.is_significant:
            ax1.text(
                0.5, max(uppers) * 1.05,
                f'p = {result.significance_test.p_value:.4f} *',
                ha='center', fontsize=12
            )

        # 改进幅度
        ax2.bar(
            ['Improvement'],
            [result.improvement * 100],
            color='green' if result.improvement > 0 else 'red'
        )
        ax2.set_title('Relative Improvement')
        ax2.set_ylabel('Improvement (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

**使用示例**:

```python
# 评估两个模型
baseline_results = evaluator.evaluate(baseline_model, test_set)
treatment_results = evaluator.evaluate(treatment_model, test_set)

# A/B 测试
ab_tester = ABTester()
result = ab_tester.compare_models(
    baseline_scores=[r.score for r in baseline_results],
    treatment_scores=[r.score for r in treatment_results],
    score_type=ScoreType.CONTINUOUS,
    baseline_name="GPT-4",
    treatment_name="GPT-4-Turbo"
)

# 打印结果
print(f"""
A/B Test Results
================
Baseline: {result.baseline_name}
  Mean: {result.baseline_ci.mean:.3f}
  95% CI: [{result.baseline_ci.lower:.3f}, {result.baseline_ci.upper:.3f}]

Treatment: {result.treatment_name}
  Mean: {result.treatment_ci.mean:.3f}
  95% CI: [{result.treatment_ci.lower:.3f}, {result.treatment_ci.upper:.3f}]

Statistical Test: {result.significance_test.test_name}
  p-value: {result.significance_test.p_value:.4f}
  Significant: {result.significance_test.is_significant}

Improvement: {result.improvement*100:+.2f}%

Recommendation: {result.recommendation}
""")

# 可视化
StatisticsVisualizer.plot_ab_test(result, save_path="ab_test_result.png")
```

**预计工作量**: 2-3 周 (1 个工程师)

---

### 4. Trajectory/Tool 评估 ⭐⭐⭐

#### 问题描述

**Google ADK 的核心能力，RM-Gallery 完全缺失**

当前 RM-Gallery 只能评估最终输出（final response），无法评估：
- Agent 的推理过程（reasoning trajectory）
- 工具调用的正确性和效率
- 中间步骤的质量
- 任务规划能力

这对于评估 Agent 系统是重大缺陷。

#### 对比框架实现

**Google ADK 的实现**:

```python
@dataclass
class Invocation:
    """Agent 的单次调用"""
    request: Request
    response: Response
    tool_calls: list[ToolCall]
    timestamp: datetime

class TrajectoryEvaluator(Evaluator):
    """轨迹评估器"""

    def evaluate_invocations(
        self,
        actual_invocations: list[Invocation],
        expected_invocations: list[Invocation],
    ) -> EvaluationResult:
        """评估完整的执行轨迹"""

        per_invocation_results = []
        for actual, expected in zip(actual_invocations, expected_invocations):
            # 评估每个步骤
            score = self._evaluate_single_invocation(actual, expected)
            per_invocation_results.append(
                PerInvocationResult(
                    actual_invocation=actual,
                    expected_invocation=expected,
                    score=score
                )
            )

        return EvaluationResult(
            overall_score=np.mean([r.score for r in per_invocation_results]),
            per_invocation_results=per_invocation_results
        )

class ToolUseQualityEvaluator:
    """工具使用质量评估"""

    def evaluate_tool_calls(
        self,
        tool_calls: list[ToolCall],
        task: str
    ) -> dict:
        """评估工具调用质量

        评估维度:
        - Correctness: 工具选择是否正确
        - Efficiency: 是否使用了最少的工具调用
        - Parameter Accuracy: 参数是否正确
        """
        return {
            "correctness": self._eval_correctness(tool_calls, task),
            "efficiency": self._eval_efficiency(tool_calls, task),
            "parameter_accuracy": self._eval_parameters(tool_calls)
        }
```

#### 影响分析

**业务影响**:
- 🚫 无法评估 Agent 的推理质量
- 🚫 无法分析失败原因（哪一步出错了）
- 🚫 无法优化工具调用策略
- 🚫 无法评估任务规划能力

**技术债务**:
- Agent 系统评估不完整
- 难以进行细粒度的调试
- 无法优化中间步骤

#### 实现建议

**Phase 1: 轨迹数据结构**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

@dataclass
class ToolCall:
    """工具调用"""
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = None

@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    thought: str  # 推理过程
    action: str  # 行动
    observation: str  # 观察结果
    tool_calls: list[ToolCall] = None

@dataclass
class AgentTrajectory:
    """Agent 执行轨迹"""
    task: str
    steps: list[ReasoningStep]
    final_answer: str
    total_time: float
    metadata: dict[str, Any] = None
```

**Phase 2: 轨迹评估器**

```python
from rm_gallery.core.reward.base import BaseReward

class TrajectoryReward(BaseReward):
    """轨迹评估 Reward Model"""

    name: str = "trajectory_reward"

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """评估 Agent 执行轨迹"""

        trajectory = self._extract_trajectory(sample)

        scores = {
            "correctness": self._eval_correctness(trajectory),
            "efficiency": self._eval_efficiency(trajectory),
            "reasoning_quality": self._eval_reasoning(trajectory),
            "tool_use_quality": self._eval_tool_use(trajectory)
        }

        # 计算综合得分
        overall_score = sum(scores.values()) / len(scores)

        return RewardResult(
            score=overall_score,
            details=scores,
            metadata={"trajectory_length": len(trajectory.steps)}
        )

    def _eval_correctness(self, trajectory: AgentTrajectory) -> float:
        """评估最终答案的正确性"""
        # 使用 LLM-as-Judge 评估最终答案
        prompt = f"""
        Task: {trajectory.task}
        Agent's Answer: {trajectory.final_answer}

        Is this answer correct? Rate from 0.0 to 1.0.
        """
        return self._llm_judge(prompt)

    def _eval_efficiency(self, trajectory: AgentTrajectory) -> float:
        """评估效率（步骤数、时间）"""
        # 惩罚过多的步骤
        optimal_steps = self._estimate_optimal_steps(trajectory.task)
        actual_steps = len(trajectory.steps)

        if actual_steps <= optimal_steps:
            return 1.0
        else:
            # 超出最优步数，分数递减
            return max(0.0, 1.0 - (actual_steps - optimal_steps) * 0.1)

    def _eval_reasoning(self, trajectory: AgentTrajectory) -> float:
        """评估推理质量"""
        step_scores = []
        for step in trajectory.steps:
            # 评估每一步的推理
            prompt = f"""
            Thought: {step.thought}
            Action: {step.action}
            Observation: {step.observation}

            Is this reasoning step logical and helpful? Rate from 0.0 to 1.0.
            """
            step_scores.append(self._llm_judge(prompt))

        return np.mean(step_scores) if step_scores else 0.0

    def _eval_tool_use(self, trajectory: AgentTrajectory) -> float:
        """评估工具使用质量"""
        all_tool_calls = []
        for step in trajectory.steps:
            if step.tool_calls:
                all_tool_calls.extend(step.tool_calls)

        if not all_tool_calls:
            return 1.0  # 不需要工具

        scores = []
        for tool_call in all_tool_calls:
            # 评估工具选择和参数
            score = self._eval_single_tool_call(tool_call, trajectory.task)
            scores.append(score)

        return np.mean(scores)

class ToolUseReward(BaseReward):
    """工具使用专项评估"""

    name: str = "tool_use_reward"

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """评估工具使用"""

        tool_calls = self._extract_tool_calls(sample)

        scores = {
            "tool_selection": self._eval_tool_selection(tool_calls),
            "parameter_correctness": self._eval_parameters(tool_calls),
            "error_handling": self._eval_error_handling(tool_calls),
            "necessity": self._eval_necessity(tool_calls)
        }

        return RewardResult(
            score=sum(scores.values()) / len(scores),
            details=scores
        )

    def _eval_tool_selection(self, tool_calls: list[ToolCall]) -> float:
        """评估工具选择是否合适"""
        correct_selections = 0
        for tool_call in tool_calls:
            if self._is_tool_appropriate(tool_call):
                correct_selections += 1

        return correct_selections / len(tool_calls) if tool_calls else 1.0

    def _eval_parameters(self, tool_calls: list[ToolCall]) -> float:
        """评估参数正确性"""
        correct_params = 0
        for tool_call in tool_calls:
            if self._are_parameters_correct(tool_call):
                correct_params += 1

        return correct_params / len(tool_calls) if tool_calls else 1.0
```

**Phase 3: ReAct 模式支持**

```python
class ReActTrajectoryParser:
    """解析 ReAct 格式的轨迹"""

    @staticmethod
    def parse_trajectory(text: str) -> AgentTrajectory:
        """解析 ReAct 输出

        Example input:
        ```
        Task: Find the capital of France

        Thought 1: I need to search for information about France's capital
        Action 1: Search["capital of France"]
        Observation 1: Paris is the capital of France

        Thought 2: I have found the answer
        Action 2: Finish["Paris"]
        ```
        """
        lines = text.strip().split('\n')
        task = ""
        steps = []
        current_step = None

        for line in lines:
            line = line.strip()

            if line.startswith("Task:"):
                task = line[5:].strip()
            elif line.startswith("Thought"):
                if current_step:
                    steps.append(current_step)
                current_step = ReasoningStep(
                    step_number=len(steps) + 1,
                    thought=line.split(":", 1)[1].strip(),
                    action="",
                    observation=""
                )
            elif line.startswith("Action") and current_step:
                action_text = line.split(":", 1)[1].strip()
                current_step.action = action_text
                # 解析工具调用
                if "[" in action_text and "]" in action_text:
                    tool_name = action_text[:action_text.index("[")]
                    tool_args = action_text[action_text.index("[")+1:action_text.index("]")]
                    current_step.tool_calls = [
                        ToolCall(
                            tool_name=tool_name,
                            arguments={"query": tool_args}
                        )
                    ]
            elif line.startswith("Observation") and current_step:
                current_step.observation = line.split(":", 1)[1].strip()

        if current_step:
            steps.append(current_step)

        return AgentTrajectory(
            task=task,
            steps=steps,
            final_answer=steps[-1].observation if steps else "",
            total_time=0.0
        )
```

**使用示例**:

```python
# 评估 Agent 轨迹
trajectory_rm = TrajectoryReward(llm_model="gpt-4")

sample = DataSample(
    query="Find the population of Tokyo",
    response="""
    Thought 1: I need to search for Tokyo's population
    Action 1: Search["Tokyo population"]
    Observation 1: Tokyo has approximately 14 million people

    Thought 2: I have the answer
    Action 2: Finish["Tokyo has approximately 14 million people"]
    """
)

result = trajectory_rm.evaluate(sample)
print(f"""
Trajectory Evaluation:
- Overall Score: {result.score:.2f}
- Correctness: {result.details['correctness']:.2f}
- Efficiency: {result.details['efficiency']:.2f}
- Reasoning Quality: {result.details['reasoning_quality']:.2f}
- Tool Use Quality: {result.details['tool_use_quality']:.2f}
""")
```

**预计工作量**: 4-5 周 (1-2 个工程师)

---

### 5. 代码评估能力 ⭐⭐

#### 问题描述

**LangChain OpenEvals 的独特优势，RM-Gallery 完全缺失**

代码生成是 LLM 的重要应用场景，但 RM-Gallery 缺乏代码评估能力：
- 无静态类型检查（Pyright, Mypy）
- 无代码执行测试
- 无安全沙箱环境
- 无代码质量评估

#### 对比框架实现

**OpenEvals 的代码评估**:

```python
from openevals.code import (
    pyright_evaluator,
    mypy_evaluator,
    sandbox_execution_evaluator,
    code_correctness_llm_evaluator
)

# 静态类型检查
pyright_eval = pyright_evaluator()
result = pyright_eval(
    inputs={"task": "Write a function to add two numbers"},
    outputs={"code": "def add(a, b): return a + b"}
)

# 沙箱执行
exec_eval = sandbox_execution_evaluator(
    test_cases=[
        {"inputs": {"a": 1, "b": 2}, "expected_output": 3},
        {"inputs": {"a": -1, "b": 1}, "expected_output": 0}
    ]
)

# LLM 评估代码质量
quality_eval = code_correctness_llm_evaluator()
```

#### 实现建议 (简化版)

```python
import subprocess
import tempfile
from pathlib import Path

class CodeReward(BaseReward):
    """代码评估 Reward Model"""

    name: str = "code_reward"

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """评估代码质量"""

        code = self._extract_code(sample.response)

        scores = {
            "syntax_valid": self._check_syntax(code),
            "type_safety": self._run_pyright(code),
            "test_pass": self._run_tests(code, sample.test_cases),
            "code_quality": self._eval_quality_with_llm(code)
        }

        return RewardResult(
            score=sum(scores.values()) / len(scores),
            details=scores
        )

    def _check_syntax(self, code: str) -> float:
        """检查语法错误"""
        try:
            compile(code, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0

    def _run_pyright(self, code: str) -> float:
        """运行 Pyright 类型检查"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ['pyright', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            # 解析输出，计算错误数
            errors = result.stdout.count('error')
            if errors == 0:
                return 1.0
            else:
                return max(0.0, 1.0 - errors * 0.1)
        finally:
            Path(temp_path).unlink()

    def _run_tests(
        self, code: str, test_cases: list[dict]
    ) -> float:
        """运行测试用例（简化版，实际应使用沙箱）"""
        if not test_cases:
            return 1.0

        # 警告：这是不安全的，仅用于演示
        # 生产环境应使用 Docker 沙箱
        passed = 0
        for test in test_cases:
            try:
                exec(code, test['inputs'])
                # 检查输出
                if 'expected_output' in test:
                    # 简化的检查逻辑
                    passed += 1
            except Exception:
                pass

        return passed / len(test_cases)
```

**预计工作量**: 2-3 周 (1 个工程师)

---

### 6-10. 其他缺失能力

由于篇幅限制，以下能力简要说明：

#### 6. RAG 专用评估 ⭐⭐

**缺失内容**:
- Correctness: 答案正确性
- Helpfulness: 答案有用性
- Groundedness: 基于检索内容
- Retrieval Relevance: 检索质量

**预计工作量**: 2 周

#### 7. 可扩展 Solver 系统 ⭐⭐

**缺失内容**:
- CompletionFn 抽象
- Solver 组合
- Postprocessor 链

**预计工作量**: 3 周

#### 8. CI/CD 集成 ⭐⭐

**缺失内容**:
- GitHub Action
- 自动化评估触发
- 报告生成

**预计工作量**: 1-2 周

#### 9. 多存储后端 ⭐

**缺失内容**:
- GCS/S3 支持
- 数据库后端
- 团队共享能力

**预计工作量**: 2 周

#### 10. 系统化 Safety 评估 ⭐⭐

**缺失内容**:
- Violence/Sexual/SelfHarm/Hate 评估器
- 间接攻击检测
- 代码漏洞检测

**预计工作量**: 3 周

---

## 优先级建议

### P0 (必须完成) - 3 个月内

这些是**所有对比框架的共性能力**，是 RM-Gallery 作为通用评估框架的基础设施：

| 能力 | 工作量 | 业务价值 | 技术价值 |
|------|--------|----------|----------|
| **Eval Case/Set 管理系统** | 3-4 周 | 高 - 系统化测试 | 高 - 基础架构 |
| **统计分析和对比** | 2-3 周 | 高 - 科学评估 | 高 - 可信度 |
| **用户模拟器** | 2-3 周 | 高 - 多轮对话 | 中 - 扩展能力 |

**预期效果**:
- ✅ 支持系统化的测试用例管理
- ✅ 提供科学严谨的评估结果
- ✅ 支持基本的多轮对话评估
- ✅ 追平其他框架的基础能力

### P1 (重要功能) - 6 个月内

这些能力让 RM-Gallery 在特定领域形成竞争优势：

| 能力 | 工作量 | 业务价值 | 差异化价值 |
|------|--------|----------|------------|
| **Trajectory/Tool 评估** | 4-5 周 | 高 - Agent 评估 | 高 - 与 ADK 竞争 |
| **RAG 专用评估** | 2 周 | 中 - 覆盖主流场景 | 中 - 与 OpenEvals 竞争 |
| **可扩展 Solver 系统** | 3 周 | 中 - 架构灵活性 | 中 - 与 OpenAI Evals 对齐 |

**预期效果**:
- ✅ 全面的 Agent 评估能力
- ✅ 覆盖 RAG 等主流应用
- ✅ 更灵活的架构设计

### P2 (增强功能) - 12 个月内

这些能力提升工程化水平和用户体验：

| 能力 | 工作量 | 业务价值 | 工程价值 |
|------|--------|----------|----------|
| **代码评估能力** | 2-3 周 | 中 - 代码生成场景 | 中 |
| **CI/CD 集成** | 1-2 周 | 中 - 开发体验 | 高 |
| **多存储后端** | 2 周 | 低 - 企业需求 | 中 |
| **系统化 Safety 评估** | 3 周 | 中 - 安全性 | 中 |

---

## 实施路线图

### Phase 1: 基础能力补齐 (Q1 2025)

**目标**: 具备通用评估框架的核心能力

**里程碑**:
1. Week 1-4: **Eval Case/Set 管理系统**
   - 数据模型设计
   - LocalManager 实现
   - YAML 配置支持
   - CLI 工具

2. Week 5-7: **统计分析模块**
   - 置信区间计算
   - 显著性检验
   - A/B 测试框架
   - 可视化报告

3. Week 8-10: **用户模拟器**
   - 基础接口设计
   - StaticUserSimulator
   - LLMUserSimulator
   - 多轮评估框架

**验收标准**:
- ✅ 可以创建和管理测试集
- ✅ 可以进行统计严谨的模型对比
- ✅ 可以自动化多轮对话评估

### Phase 2: Agent 评估能力 (Q2 2025)

**目标**: 支持全面的 Agent 系统评估

**里程碑**:
1. Week 1-3: **轨迹数据结构**
   - AgentTrajectory 定义
   - ReActParser 实现
   - 数据采集 hooks

2. Week 4-7: **Trajectory 评估器**
   - TrajectoryReward 实现
   - 推理质量评估
   - 效率评估
   - 工具使用评估

3. Week 8-10: **集成和优化**
   - 与现有 RM 集成
   - 性能优化
   - 文档和示例

**验收标准**:
- ✅ 可以评估 Agent 执行轨迹
- ✅ 可以分析工具使用质量
- ✅ 可以评估推理过程

### Phase 3: 应用场景扩展 (Q3 2025)

**目标**: 覆盖主流 LLM 应用场景

**里程碑**:
1. Week 1-2: **RAG 评估器**
   - Correctness/Helpfulness/Groundedness
   - Retrieval Relevance

2. Week 3-5: **代码评估能力**
   - 语法检查
   - 类型检查
   - 简化的测试执行

3. Week 6-8: **Solver 抽象层**
   - CompletionFn 接口
   - Solver 基类
   - Postprocessor 链

**验收标准**:
- ✅ 支持 RAG 应用评估
- ✅ 支持代码生成评估
- ✅ 更灵活的评估架构

### Phase 4: 工程化提升 (Q4 2025)

**目标**: 提升工程化和团队协作能力

**里程碑**:
1. Week 1-2: **CI/CD 集成**
   - GitHub Action
   - 自动化评估
   - 报告生成

2. Week 3-4: **多存储后端**
   - GCS/S3 支持
   - 数据库后端

3. Week 5-7: **Safety 评估器**
   - 四大类安全评估
   - 间接攻击检测

**验收标准**:
- ✅ 可集成到 CI/CD 流程
- ✅ 支持团队协作
- ✅ 全面的安全评估

---

## 差异化定位建议

### RM-Gallery 的核心优势

RM-Gallery 在以下方面具有**独特优势**，应该继续保持和加强：

1. **Reward Model 专业性** ⭐⭐⭐
   - Rubric 生成和管理
   - Rubric-Critic-Score 范式
   - 丰富的内置 RM Gallery

2. **Training 集成** ⭐⭐⭐
   - 与 VERL 等框架集成
   - RM 训练流程
   - 闭环优化能力

3. **高性能 Serving** ⭐⭐
   - 高吞吐量
   - 容错机制
   - 批处理优化

### 建议的差异化定位

> **RM-Gallery: 专业的 Reward Model 平台 + 完整的 LLM 评估框架**

**核心价值主张**:
1. **深度**: Reward Model 领域的最专业工具
2. **广度**: 覆盖 LLM 评估的全生命周期
3. **闭环**: 从评估到训练到应用的完整链路

### 竞争策略

**vs OpenAI Evals**:
- ✅ 更专注于 Reward Model
- ✅ 更好的 Training 集成
- ⚠️ 需要补齐 Registry 和 Solver 系统

**vs OpenEvals**:
- ✅ 更系统的 RM 管理
- ✅ 更强的 Training 能力
- ⚠️ 需要补齐 RAG 和代码评估

**vs Google ADK**:
- ✅ 更专注于评估（不是 Agent 开发）
- ✅ 更丰富的 RM Gallery
- ⚠️ 需要补齐 Trajectory 和用户模拟器

**vs Azure AI Evals**:
- ✅ 更灵活（不绑定 Azure）
- ✅ 更专业的 RM 能力
- ⚠️ 需要补齐统计分析和 CI/CD

### 目标用户画像

**Primary Users**:
1. **RLHF/RLAIF 研究者** - 需要训练和评估 RM
2. **LLM 应用开发者** - 需要评估 LLM 输出质量
3. **Agent 开发者** - 需要评估 Agent 系统

**Use Cases**:
- ✅ Reward Model 训练和评估
- ✅ LLM 输出质量评估
- ✅ Agent 系统评估
- ✅ RLHF/RLAIF 流程
- ✅ Model-as-Judge 应用

---

## 总结和建议

### 关键发现

1. **基础能力缺失**: Eval Case/Set 管理、统计分析、用户模拟器是所有框架的共性，RM-Gallery 必须补齐

2. **Agent 评估不足**: Trajectory 和 Tool 评估是 Agent 时代的刚需，需要尽快支持

3. **应用场景覆盖**: RAG、代码评估等主流场景需要专门的评估器

4. **工程化程度**: CI/CD 集成、多存储后端等工程化能力有待提升

### 战略建议

**短期 (3 个月)**:
- 🎯 **补齐基础能力** - P0 三项必须完成
- 🎯 **保持核心优势** - 继续深化 Reward Model 能力

**中期 (6 个月)**:
- 🎯 **Agent 评估能力** - Trajectory/Tool 评估
- 🎯 **应用场景扩展** - RAG、代码评估

**长期 (12 个月)**:
- 🎯 **工程化提升** - CI/CD、团队协作
- 🎯 **生态建设** - 与主流框架集成

### 成功指标

**功能完整性**:
- ✅ 覆盖 80% 的对比框架核心能力
- ✅ 至少 5 个应用场景的专用评估器
- ✅ 完整的 Agent 评估能力

**易用性**:
- ✅ 5 分钟快速上手
- ✅ 完善的文档和示例
- ✅ CLI 和 API 双接口

**性能和可靠性**:
- ✅ 统计学严谨性
- ✅ 高吞吐量评估
- ✅ 可扩展架构

---

## 附录

### A. 对比框架资源

- **OpenAI Evals**: https://github.com/openai/evals
- **LangChain OpenEvals**: https://github.com/langchain-ai/openevals
- **Google ADK**: https://github.com/google/adk-python
- **Azure AI Agent Evals**: https://github.com/microsoft/ai-agent-evals

### B. 参考文献

1. McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages"
2. Agresti, A. (2013). "Categorical Data Analysis" (3rd ed.)
3. Wilson, E. B. (1927). "Probable Inference, the Law of Succession, and Statistical Inference"

### C. 术语表

- **Eval Case**: 单个测试用例
- **Eval Set**: 测试集合
- **Trajectory**: Agent 执行轨迹
- **User Simulator**: 用户模拟器
- **Confidence Interval**: 置信区间
- **Statistical Significance**: 统计显著性
- **McNemar's Test**: 配对二元数据的显著性检验

---

**文档结束**

