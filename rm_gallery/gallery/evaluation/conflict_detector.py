"""
Conflict Detector
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Dict, List, Optional, Type

import fire
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.base import BaseLLMReward, BasePairWiseReward
from rm_gallery.core.reward.evaluator import BaseEvaluator
from rm_gallery.core.reward.schema import RewardDimensionWithRank, RewardResult
from rm_gallery.core.utils.file import write_json
from rm_gallery.gallery.evaluation.template import (
    PairComparisonTemplate,
    PointwiseTemplate,
)


class ConflictType(str, Enum):
    """Conflict type enumeration for pairwise comparison analysis.

    Attributes:
        SYMMETRY: Symmetry conflict when A>B and B>A
        TRANSITIVITY: Transitivity conflict when A>B>C but A not>C
        CYCLE: Cycle conflict when circular preferences exist
    """

    SYMMETRY = "symmetry"  # Symmetry conflict
    TRANSITIVITY = "transitivity"  # Transitivity conflict
    CYCLE = "cycle"  # Cycle conflict


class Conflict(BaseModel):
    """Conflict record for storing detected inconsistencies.

    Attributes:
        conflict_type: Type of conflict detected
        involved_items: List of response indices involved in conflict
        description: Human-readable conflict description
        severity: Numerical severity score (default=1.0)
    """

    conflict_type: ConflictType = Field(default=..., description="Conflict type")
    involved_items: List[int] = Field(default=..., description="Involved items")
    description: str = Field(default=..., description="Conflict description")
    severity: float = Field(default=1.0, description="Conflict severity")


class ComparisonResult(Enum):
    """比较结果枚举"""

    A_BETTER = 1  # A > B
    B_BETTER = -1  # A < B
    TIE = 0  # A = B


@dataclass
class ConflictMetrics:
    """基于强连通分量（SCC）检测的冲突指标（百分比形式，越低越好）"""

    overall_conflict_rate: float  # 总体冲突率 (%)
    transitivity_conflict_rate: float  # 传递性冲突率 (%) - 等同于总体冲突率（SCC检测统一处理）
    cycle_conflict_rate: float  # 循环冲突率 (%) - 等同于总体冲突率（SCC检测统一处理）
    conflict_density_rate: float  # 冲突密度率 - 平均每个样本的SCC冲突节点数量
    conflicts_per_comparison: float  # 每次比较的平均冲突节点数

    # 保留用于详细分析的原始数据
    total_samples: int
    total_conflicts: int  # 总SCC冲突节点数
    total_comparisons: int  # 总成功比较次数


class ConflictDetector:
    """冲突检测器核心类，完全移植自原始算法"""

    def __init__(self):
        pass

    def build_comparison_matrix(
        self, responses: List[str], comparison_results: Dict[tuple, ComparisonResult]
    ) -> np.ndarray:
        """
        构建比较矩阵

        Args:
            responses: 响应列表
            comparison_results: 比较结果字典 {(i,j): ComparisonResult}

        Returns:
            比较矩阵 M[i][j] = 1表示i>j, -1表示i<j, 0表示i=j
        """
        n = len(responses)
        matrix = np.zeros((n, n), dtype=int)

        for (i, j), result in comparison_results.items():
            matrix[i][j] = result.value
            matrix[j][i] = -result.value  # 对称填充

        return matrix

    def detect_symmetry_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """检测对称性冲突（在当前单次比较场景下，此类冲突不存在）"""
        # 在我们的实现中，每对响应只比较一次，然后自动填充对称位置
        # 因此不会有真正的对称性冲突，返回空列表
        return []

    def detect_transitivity_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """检测传递性冲突"""
        conflicts = []
        n = matrix.shape[0]

        # 添加调试信息
        total_checks = 0
        valid_chains = 0

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        total_checks += 1
                        # 检查传递性: 如果 i>j 且 j>k，则应该 i>k
                        if matrix[i][j] > 0 and matrix[j][k] > 0:
                            valid_chains += 1
                            if matrix[i][k] <= 0:
                                conflicts.append(
                                    Conflict(
                                        conflict_type=ConflictType.TRANSITIVITY,
                                        involved_items=[i, j, k],
                                        description=f"传递性冲突: 响应{i}>响应{j}>响应{k}，但响应{i}{'<' if matrix[i][k] < 0 else '='}响应{k}",
                                    )
                                )

        # 只在有比较矩阵时输出调试信息
        if total_checks > 0:
            logger.debug(
                f"传递性检查: 总检查数={total_checks}, 有效链={valid_chains}, 冲突数={len(conflicts)}"
            )
            logger.debug(f"比较矩阵:\n{matrix}")

        return conflicts

    def detect_cycles(self, matrix: np.ndarray) -> List[Conflict]:
        """使用DFS检测循环冲突"""
        conflicts = []
        n = matrix.shape[0]

        def dfs_cycle_detection(node: int, path: List[int], visited: set) -> bool:
            """DFS检测环"""
            if node in path:
                # 找到环
                cycle_start = path.index(node)
                cycle_nodes = path[cycle_start:] + [node]
                if len(cycle_nodes) > 2:  # 至少3个节点的环
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.CYCLE,
                            involved_items=cycle_nodes[:-1],  # 去除重复的最后一个节点
                            description=f"循环冲突: 响应 {' > '.join(map(str, cycle_nodes))} 形成环",
                            severity=len(cycle_nodes) - 1,
                        )
                    )
                return True

            if node in visited:
                return False

            visited.add(node)
            path.append(node)

            # 访问所有子节点（i>j且matrix[i][j]>0的j）
            for next_node in range(n):
                if matrix[node][next_node] > 0:
                    if dfs_cycle_detection(next_node, path, visited):
                        return True

            path.pop()
            return False

        # 从每个节点开始检测环
        for start_node in range(n):
            visited = set()
            dfs_cycle_detection(start_node, [], visited)

        # 去重（同一个环可能被多次检测到）
        unique_conflicts = []
        seen_cycles = set()
        for conflict in conflicts:
            cycle_signature = tuple(sorted(conflict.involved_items))
            if cycle_signature not in seen_cycles:
                seen_cycles.add(cycle_signature)
                unique_conflicts.append(conflict)

        return unique_conflicts

    def has_conflicts_scc_detection(self, matrix: np.ndarray) -> bool:
        """
        使用强连通分量（SCC）检测是否存在冲突
        基于定义：存在大小>1的强连通分量即表明存在环路冲突

        Args:
            matrix: 比较矩阵 M[i][j] = 1表示i>j, -1表示i<j, 0表示i=j

        Returns:
            True如果存在冲突，False否则
        """
        n = matrix.shape[0]
        if n < 2:
            return False  # 少于2个节点无法形成环

        # 构建有向图邻接表
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i][j] > 0:
                    graph[i].append(j)  # i -> j

        # 使用Tarjan算法检测强连通分量
        sccs = self._tarjan_scc(graph)

        # 检查是否存在大小>1的强连通分量
        for scc in sccs:
            if len(scc) > 1:
                return True

        return False

    def _tarjan_scc(self, graph: List[List[int]]) -> List[List[int]]:
        """
        Tarjan算法实现强连通分量检测

        Args:
            graph: 邻接表表示的有向图

        Returns:
            强连通分量列表，每个分量是节点索引的列表
        """
        n = len(graph)
        index = 0
        stack = []
        indices = [-1] * n  # 节点的访问顺序
        lowlinks = [-1] * n  # 节点能回溯到的最早祖先
        on_stack = [False] * n
        sccs = []

        def strongconnect(v):
            nonlocal index
            # 设置节点v的深度索引
            indices[v] = index
            lowlinks[v] = index
            index += 1
            stack.append(v)
            on_stack[v] = True

            # 遍历v的所有后继节点
            for w in graph[v]:
                if indices[w] == -1:
                    # 后继节点w尚未访问，递归访问
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack[w]:
                    # 后继节点w在栈中，说明找到了一个环
                    lowlinks[v] = min(lowlinks[v], indices[w])

            # 如果v是强连通分量的根节点
            if lowlinks[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        # 对所有未访问的节点执行DFS
        for v in range(n):
            if indices[v] == -1:
                strongconnect(v)

        return sccs

    def count_scc_conflicts(self, matrix: np.ndarray) -> int:
        """
        计算存在环路冲突的强连通分量数量

        Args:
            matrix: 比较矩阵

        Returns:
            大小>1的强连通分量数量（即环路冲突数量）
        """
        n = matrix.shape[0]
        if n < 2:
            return 0

        # 构建有向图邻接表
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i][j] > 0:
                    graph[i].append(j)  # i -> j

        # 使用Tarjan算法检测强连通分量
        sccs = self._tarjan_scc(graph)

        # 统计大小>1的强连通分量数量
        conflict_count = sum(1 for scc in sccs if len(scc) > 1)

        return conflict_count

    def detect_all_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """
        检测所有类型的冲突
        现在使用强连通分量（SCC）检测方法
        """
        conflicts = []

        # 使用新的SCC方法检测冲突
        if self.has_conflicts_scc_detection(matrix):
            # 如果存在冲突，获取具体的强连通分量信息
            n = matrix.shape[0]
            graph = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j and matrix[i][j] > 0:
                        graph[i].append(j)

            sccs = self._tarjan_scc(graph)
            conflict_sccs = [scc for scc in sccs if len(scc) > 1]

            # 为每个冲突的强连通分量创建一个冲突记录
            for idx, scc in enumerate(conflict_sccs):
                # 构造环路描述
                if len(scc) == 2:
                    # 双节点环（相互偏好冲突）
                    description = f"双向冲突: 响应{scc[0]}和响应{scc[1]}之间存在相互偏好"
                else:
                    # 多节点环
                    description = f"循环冲突: 响应 {' -> '.join(map(str, scc + [scc[0]]))} 形成{len(scc)}节点环路"

                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.CYCLE,
                        involved_items=scc,
                        description=description,
                        severity=float(len(scc)),  # 环路大小作为严重程度
                    )
                )

        return conflicts


def extract_rewardbench2_responses(sample: DataSample) -> List[str]:
    """从RewardBench2样本中提取responses（1个chosen + 3个rejected）"""
    responses = []

    # 从output中提取所有响应内容
    for output in sample.output:
        responses.append(output.answer.content)

    # 确保我们有至少2个响应进行比较
    if len(responses) < 2:
        # 如果响应不足，用重复的响应填充
        while len(responses) < 2:
            responses.append(responses[0] if responses else "No response")

    return responses


def extract_prompt(sample: DataSample) -> str:
    """从样本中提取prompt"""
    if sample.input and len(sample.input) > 0:
        return sample.input[-1].content
    return ""


def create_comparison_pairs(responses: List[str]) -> List[tuple]:
    """生成所有两两比较对的索引"""
    pairs = []
    n = len(responses)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


class ConflictDetectionReward(BaseLLMReward, BasePairWiseReward):
    """支持冲突检测的奖励模型，支持pairwise和pointwise两种比较模式"""

    model_config = {"arbitrary_types_allowed": True}

    comparison_mode: str = Field(
        default="pairwise", description="比较模式: 'pairwise' 或 'pointwise'"
    )
    pointwise_template: Type[PointwiseTemplate] = Field(
        default=PointwiseTemplate, description="pointwise评分模板"
    )
    pairwise_template: Type[PairComparisonTemplate] = Field(
        default=PairComparisonTemplate, description="pairwise比较模板"
    )
    conflict_detector: ConflictDetector = Field(
        default_factory=ConflictDetector, description="冲突检测器实例"
    )
    save_detailed_outputs: bool = Field(default=True, description="是否保存详细的模型输出记录")

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """评估样本并检测冲突"""
        assert self.llm is not None

        try:
            if self.comparison_mode == "pointwise":
                return self._evaluate_pointwise_mode(sample, **kwargs)
            else:
                return self._evaluate_pairwise_mode(sample, **kwargs)
        except Exception as e:
            logger.error(f"样本评估失败: {str(e)}")
            return RewardResult(
                name=self.name, details=[], extra_data={"error": str(e)}
            )

    def _evaluate_pairwise_mode(self, sample: DataSample, **kwargs) -> RewardResult:
        """pairwise模式：直接两两比较"""
        prompt = extract_prompt(sample)
        responses = extract_rewardbench2_responses(sample)

        if len(responses) < 2:
            logger.warning("响应数量不足，跳过评估")
            return RewardResult(
                name=self.name,
                details=[],
                extra_data={"error": "insufficient_responses"},
            )

        # 获取所有比较对
        comparison_pairs = create_comparison_pairs(responses)
        comparison_results = {}
        detailed_comparisons = []

        # 执行所有两两比较
        for i, j in comparison_pairs:
            try:
                # 使用pairwise模板进行比较
                comparison_prompt = self.pairwise_template.format(
                    query=prompt, answers=[responses[i], responses[j]]
                )

                # 调用LLM
                response_text = self.llm.simple_chat(query=comparison_prompt)

                # 解析结果
                parsed = self.pairwise_template.parse(response_text)

                # 转换为ComparisonResult
                if parsed.best_answer.lower() == "a":
                    result = ComparisonResult.A_BETTER
                elif parsed.best_answer.lower() == "b":
                    result = ComparisonResult.B_BETTER
                else:
                    result = ComparisonResult.TIE

                comparison_results[(i, j)] = result

                # 保存详细信息
                if self.save_detailed_outputs:
                    detailed_comparisons.append(
                        {
                            "pair": (i, j),
                            "prompt": comparison_prompt,
                            "response": response_text,
                            "parsed_result": parsed.best_answer,
                            "reasoning": parsed.reasoning,
                        }
                    )

            except Exception as e:
                logger.warning(f"比较失败 ({i}, {j}): {e}")
                comparison_results[(i, j)] = ComparisonResult.TIE

        return self._process_comparisons_and_detect_conflicts(
            responses, comparison_results, detailed_comparisons
        )

    def _evaluate_pointwise_mode(self, sample: DataSample, **kwargs) -> RewardResult:
        """pointwise模式：独立评分后比较"""
        prompt = extract_prompt(sample)
        responses = extract_rewardbench2_responses(sample)

        if len(responses) < 2:
            logger.warning("响应数量不足，跳过评估")
            return RewardResult(
                name=self.name,
                details=[],
                extra_data={"error": "insufficient_responses"},
            )

        # 获取所有比较对
        comparison_pairs = create_comparison_pairs(responses)
        comparison_results = {}
        detailed_comparisons = []

        # 对每个比较对，并行对两个响应进行评分
        for i, j in comparison_pairs:
            try:
                # 并行评分两个响应，提高效率
                score_i, score_j = self._score_pair_parallel(
                    prompt, responses[i], responses[j]
                )

                # 检查评分是否成功
                if score_i is not None and score_j is not None:
                    # 比较分数确定偏序
                    if score_i > score_j:
                        result = ComparisonResult.A_BETTER
                    elif score_j > score_i:
                        result = ComparisonResult.B_BETTER
                    else:
                        result = ComparisonResult.TIE

                    comparison_results[(i, j)] = result

                    # 保存详细信息
                    if self.save_detailed_outputs:
                        detailed_comparisons.append(
                            {
                                "pair": (i, j),
                                "score_a": score_i,
                                "score_b": score_j,
                                "result": result.name,
                            }
                        )
                else:
                    # 评分失败，跳过该比较对
                    logger.warning(
                        f"pointwise评分失败，跳过比较对 ({i}, {j}): Score A: {score_i}, Score B: {score_j}"
                    )

            except Exception as e:
                logger.warning(f"pointwise比较失败 ({i}, {j}): {e}")

        return self._process_comparisons_and_detect_conflicts(
            responses, comparison_results, detailed_comparisons
        )

    def _score_single_response(self, prompt: str, response: str) -> Optional[float]:
        """
        使用pointwise模板对单个响应进行评分

        Args:
            prompt: 原始问题
            response: 要评分的响应

        Returns:
            评分（1-10分），失败时返回None
        """
        try:
            # 使用pointwise模板生成评分prompt
            scoring_prompt = self.pointwise_template.format(
                query=prompt, response=response
            )

            # 调用模型进行评分
            model_response = self.llm.simple_chat(query=scoring_prompt)

            # 解析评分结果
            parsed = self.pointwise_template.parse(model_response)

            # 检查评分是否在有效范围内
            if 1.0 <= parsed.score <= 10.0:
                return parsed.score
            else:
                logger.warning(f"评分超出范围: {parsed.score}")
                return None

        except Exception as e:
            logger.warning(f"评分失败: {e}")
            return None

    def _score_pair_parallel(
        self, prompt: str, response_a: str, response_b: str
    ) -> tuple[Optional[float], Optional[float]]:
        """
        并行评分一对响应，提高效率

        Args:
            prompt: 原始问题
            response_a: 第一个响应
            response_b: 第二个响应

        Returns:
            (score_a, score_b) 元组，失败时对应位置为None
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 同时提交两个评分任务
            future_a = executor.submit(self._score_single_response, prompt, response_a)
            future_b = executor.submit(self._score_single_response, prompt, response_b)

            # 等待两个任务完成
            score_a = future_a.result()
            score_b = future_b.result()

            return score_a, score_b

    def _process_comparisons_and_detect_conflicts(
        self,
        responses: List[str],
        comparison_results: Dict[tuple, ComparisonResult],
        detailed_comparisons: List[Dict],
    ) -> RewardResult:
        """处理比较结果并检测冲突"""

        # 构建比较矩阵
        matrix = self.conflict_detector.build_comparison_matrix(
            responses, comparison_results
        )

        # 检测冲突
        conflicts = self.conflict_detector.detect_all_conflicts(matrix)

        # 计算统计信息
        expected_comparisons = len(create_comparison_pairs(responses))
        successful_comparisons = len(comparison_results)

        # 计算冲突统计
        conflict_types = {ct.value: 0 for ct in ConflictType}
        for conflict in conflicts:
            conflict_types[conflict.conflict_type.value] += 1

        # 构建extra_data
        extra_data = {
            "responses": responses,
            "comparison_matrix": matrix.tolist(),
            "comparison_results": {
                f"{i}-{j}": result.name for (i, j), result in comparison_results.items()
            },
            "comparison_quality": {
                "expected_comparisons": expected_comparisons,
                "successful_comparisons": successful_comparisons,
                "failed_comparisons": expected_comparisons - successful_comparisons,
                "success_rate": successful_comparisons / expected_comparisons
                if expected_comparisons > 0
                else 0.0,
            },
            "conflicts": [
                {
                    "type": c.conflict_type.value,
                    "involved_items": c.involved_items,
                    "description": c.description,
                    "severity": c.severity,
                }
                for c in conflicts
            ],
            "total_conflicts": len(conflicts),
            "conflict_types": conflict_types,
            "comparison_mode": self.comparison_mode,
        }

        # 添加详细输出
        if self.save_detailed_outputs and detailed_comparisons:
            extra_data["detailed_comparisons"] = detailed_comparisons

        # 创建简单的rank分数（用于兼容性）
        scores = [0.0] * len(responses)
        if len(conflicts) == 0:
            # 无冲突时，给第一个响应较高分数
            scores[0] = 1.0

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=f"{self.name}_conflict_detection",
                    reason=f"检测到 {len(conflicts)} 个冲突，成功比较 {successful_comparisons}/{expected_comparisons}",
                    rank=scores,
                )
            ],
            extra_data=extra_data,
        )

    def _parallel(self, func, sample, thread_pool=None, **kwargs) -> DataSample:
        """精简的并行处理方法，支持 pairwise 和 pointwise 模式"""
        sample = sample.model_copy(deep=True)

        # 提取数据
        prompt = extract_prompt(sample)
        responses = extract_rewardbench2_responses(sample)

        if len(responses) < 2:
            sample.input[-1].additional_kwargs = {
                "conflict_detector": {"comparison_results": {}}
            }
            return sample

        # 生成比较对
        pairs = create_comparison_pairs(responses)
        comparison_results = {}

        # 根据模式进行比较
        for i, j in pairs:
            try:
                if self.comparison_mode == "pointwise":
                    # pointwise: 并行评分后比较，提高效率
                    score_a, score_b = self._score_pair_parallel(
                        prompt, responses[i], responses[j]
                    )
                    # 处理评分失败的情况，使用默认分数5.0
                    if score_a is None:
                        score_a = 5.0
                    if score_b is None:
                        score_b = 5.0
                    comparison_results[(i, j)] = (
                        1 if score_a > score_b else (-1 if score_b > score_a else 0)
                    )
                else:
                    # pairwise: 直接比较
                    comparison_prompt = self.pairwise_template.format(
                        query=prompt, answers=[responses[i], responses[j]]
                    )
                    response_text = self.llm.simple_chat(comparison_prompt)
                    parsed_result = self.pairwise_template.parse(response_text)

                    if parsed_result.best_answer.lower() == "a":
                        comparison_results[(i, j)] = 1
                    elif parsed_result.best_answer.lower() == "b":
                        comparison_results[(i, j)] = -1
                    else:
                        comparison_results[(i, j)] = 0
            except Exception as e:
                logger.warning(f"比较失败 ({i}, {j}): {e}")
                comparison_results[(i, j)] = 0

        # 存储结果
        sample.input[-1].additional_kwargs = {
            "conflict_detector": {"comparison_results": comparison_results}
        }
        return sample

    def _score_response(self, prompt: str, response: str) -> float:
        """为 pointwise 模式评分单个响应"""
        try:
            scoring_prompt = self.pointwise_template.format(
                query=prompt, response=response
            )
            model_response = self.llm.simple_chat(scoring_prompt)
            parsed_result = self.pointwise_template.parse(model_response)
            return parsed_result.score
        except Exception as e:
            logger.warning(f"评分失败: {e}")
            return 5.0


class ConflictDetectionEvaluator(BaseEvaluator):
    """冲突检测评估器，支持多线程处理并计算冲突指标"""

    reward: ConflictDetectionReward = Field(default=..., description="冲突检测奖励模型")

    def _detect_ties_mode(self, sample: DataSample) -> bool:
        """检测是否为Ties子集"""
        if hasattr(sample, "metadata") and sample.metadata:
            subset = sample.metadata.get("subset", "").lower()
            return subset == "ties"
        return False

    def _calculate_accuracy_for_sample(
        self, sample: DataSample, comparison_results: Dict[tuple, int]
    ) -> Dict:
        """计算单个样本的准确率：只考虑chosen vs rejected的比较结果"""
        responses = extract_rewardbench2_responses(sample)
        n = len(responses)

        # 找到chosen和rejected响应的索引
        chosen_index = None
        rejected_indices = []

        for i, output in enumerate(sample.output):
            if hasattr(output.answer, "label") and isinstance(
                output.answer.label, dict
            ):
                preference = output.answer.label.get("preference")
                if preference == "chosen":
                    chosen_index = i
                elif preference == "rejected":
                    rejected_indices.append(i)

        if chosen_index is None:
            return {
                "is_correct": False,
                "chosen_index": None,
                "rejected_indices": rejected_indices,
                "chosen_vs_rejected_comparisons": 0,
                "chosen_wins": 0,
                "chosen_losses": 0,
                "chosen_ties": 0,
                "accuracy": 0.0,
                "chosen_dominance": 0,
                "error": "No chosen response found",
            }

        if not rejected_indices:
            return {
                "is_correct": False,
                "chosen_index": chosen_index,
                "rejected_indices": [],
                "chosen_vs_rejected_comparisons": 0,
                "chosen_wins": 0,
                "chosen_losses": 0,
                "chosen_ties": 0,
                "accuracy": 0.0,
                "chosen_dominance": 0,
                "error": "No rejected responses found",
            }

        # 统计chosen vs rejected的比较结果
        chosen_wins = 0
        chosen_losses = 0
        chosen_ties = 0
        total_chosen_vs_rejected = 0

        for (i, j), result in comparison_results.items():
            # 检查是否是chosen vs rejected的比较
            is_chosen_vs_rejected = False
            chosen_better = False

            if i == chosen_index and j in rejected_indices:
                is_chosen_vs_rejected = True
                chosen_better = result > 0  # chosen > rejected
            elif j == chosen_index and i in rejected_indices:
                is_chosen_vs_rejected = True
                chosen_better = result < 0  # rejected < chosen (即chosen > rejected)

            if is_chosen_vs_rejected:
                total_chosen_vs_rejected += 1
                if result > 0 and i == chosen_index:  # chosen > rejected
                    chosen_wins += 1
                elif result < 0 and j == chosen_index:  # rejected < chosen
                    chosen_wins += 1
                elif result < 0 and i == chosen_index:  # chosen < rejected
                    chosen_losses += 1
                elif result > 0 and j == chosen_index:  # rejected > chosen
                    chosen_losses += 1
                else:  # result == 0, tie
                    chosen_ties += 1

        # 计算准确率：chosen获胜的比例
        accuracy = (
            chosen_wins / total_chosen_vs_rejected
            if total_chosen_vs_rejected > 0
            else 0.0
        )

        # 计算chosen主导度：检查chosen是否是唯一胜利数最高的
        chosen_dominance = 0
        if comparison_results:  # 只有在有比较结果时才计算
            # 统计每个响应的总胜利次数
            win_counts = [0] * n
            for (i, j), result in comparison_results.items():
                if result > 0:  # i > j
                    win_counts[i] += 1
                elif result < 0:  # i < j
                    win_counts[j] += 1

            # 检查chosen是否是唯一的最高胜利数
            if chosen_index is not None:
                chosen_win_count = win_counts[chosen_index]
                max_win_count = max(win_counts)

                # chosen是唯一胜利数最高的情况
                if (
                    chosen_win_count == max_win_count
                    and win_counts.count(max_win_count) == 1
                ):
                    chosen_dominance = 1

        return {
            "is_correct": accuracy > 0.5,  # 如果chosen获胜超过一半的比较，认为正确
            "chosen_index": chosen_index,
            "rejected_indices": rejected_indices,
            "chosen_vs_rejected_comparisons": total_chosen_vs_rejected,
            "chosen_wins": chosen_wins,
            "chosen_losses": chosen_losses,
            "chosen_ties": chosen_ties,
            "accuracy": accuracy,
            "chosen_dominance": chosen_dominance,
            "strategy": "chosen_vs_rejected_only",
        }

    def _evaluate_single_sample(self, sample: DataSample, **kwargs) -> DataSample:
        """评估单个样本 - 用于并行处理"""
        try:
            # 随机打乱响应顺序以避免位置偏差
            # 使用样本内容的hash作为种子，确保相同样本在不同模型间有相同的打乱结果
            import hashlib
            import random

            sample_copy = sample.model_copy(deep=True)

            # 基于样本内容创建确定性的种子
            sample_hash = hashlib.md5(str(sample.input).encode()).hexdigest()
            sample_seed = int(sample_hash[:8], 16)  # 使用hash的前8位作为种子

            # 临时设置随机种子进行打乱
            random_state = random.getstate()  # 保存当前随机状态
            random.seed(sample_seed)
            random.shuffle(sample_copy.output)
            random.setstate(random_state)  # 恢复之前的随机状态

            # 使用 _parallel 方法处理样本中的所有比较对
            processed_sample = self.reward._parallel(
                func=self.reward._evaluate,
                sample=sample_copy,
                thread_pool=None,  # 让 _parallel 创建自己的线程池
                **kwargs,
            )

            # 分析冲突 - 内联处理
            if "conflict_detector" in processed_sample.input[-1].additional_kwargs:
                conflict_data = processed_sample.input[-1].additional_kwargs[
                    "conflict_detector"
                ]
                comparison_results = conflict_data.get("comparison_results", {})

                if comparison_results:
                    responses = extract_rewardbench2_responses(sample_copy)
                    n = len(responses)

                    # 构建比较矩阵
                    comparison_matrix = np.zeros((n, n), dtype=int)
                    for (i, j), score in comparison_results.items():
                        comparison_matrix[i][j] = score
                        comparison_matrix[j][i] = -score

                    # 检测冲突
                    conflicts = self.reward.conflict_detector.detect_all_conflicts(
                        comparison_matrix
                    )
                    conflict_types = {
                        ct.value: sum(1 for c in conflicts if c.conflict_type == ct)
                        for ct in ConflictType
                    }

                    # 更新冲突数据
                    conflict_data["conflicts"] = [
                        {
                            "type": c.conflict_type.value,
                            "involved_items": c.involved_items,
                            "description": c.description,
                            "severity": c.severity,
                        }
                        for c in conflicts
                    ]
                    conflict_data["conflict_types"] = conflict_types

            # 计算准确率：胜利次数最多的是否是chosen（注意使用打乱后的sample_copy）
            if "conflict_detector" in processed_sample.input[-1].additional_kwargs:
                conflict_data = processed_sample.input[-1].additional_kwargs[
                    "conflict_detector"
                ]
                comparison_results = conflict_data.get("comparison_results", {})
                accuracy_data = self._calculate_accuracy_for_sample(
                    sample_copy, comparison_results
                )
            else:
                accuracy_data = {"error": "No comparison results found"}

            # 将处理结果存储在样本元数据中
            sample.metadata = sample.metadata or {}
            sample.metadata["conflict_evaluation_result"] = {
                "comparison_results": processed_sample.input[-1]
                .additional_kwargs.get("conflict_detector", {})
                .get("comparison_results", {}),
                "conflicts": processed_sample.input[-1]
                .additional_kwargs.get("conflict_detector", {})
                .get("conflicts", []),
                "conflict_types": processed_sample.input[-1]
                .additional_kwargs.get("conflict_detector", {})
                .get("conflict_types", {}),
                "accuracy_data": accuracy_data,
                "extra_data": {
                    "total_responses": len(sample.output),
                    "comparison_mode": self.reward.comparison_mode,
                },
            }

            return sample
        except Exception as e:
            logger.error(f"样本评估失败: {str(e)}")
            # 返回带有错误信息的样本
            sample.metadata = sample.metadata or {}
            sample.metadata["conflict_evaluation_error"] = str(e)
            return sample

    def _parallel_evaluate(
        self, samples: List[DataSample], desc: str, max_workers: int = 8, **kwargs
    ) -> List[DataSample]:
        """并行评估，与rewardbench2.py风格保持一致"""
        results = [None] * len(samples)
        completed_count = 0

        def update_progress_bar(done, total):
            """简单的进度指示器"""
            progress = int(50 * done / total) if total > 0 else 0
            print(
                f"\r[{'#' * progress}{'.' * (50 - progress)}] {done}/{total}",
                end="",
                flush=True,
            )

        # 创建带有kwargs的评估函数
        eval_func = partial(self._evaluate_single_sample, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务并映射到原始索引
            future_to_index = {
                executor.submit(eval_func, sample): i
                for i, sample in enumerate(samples)
            }

            # 处理完成的任务
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=60)  # Add a timeout of 60 seconds
                    results[index] = result
                except Exception as e:
                    logger.error(f"任务失败，样本索引 {index}: {str(e)}")
                    # 创建错误结果
                    sample = samples[index]
                    sample.metadata = sample.metadata or {}
                    sample.metadata["conflict_evaluation_error"] = str(e)
                    results[index] = sample

                completed_count += 1
                update_progress_bar(completed_count, len(samples))

        print()  # Ensure the executor is properly shut down
        executor.shutdown(wait=True)  # Explicitly shut down the executor
        return results

    def run(self, samples: List[DataSample], max_workers: int = 8, **kwargs) -> dict:
        """执行评估，支持并行处理"""
        if not samples:
            return {"error": "没有样本需要评估"}

        # 分离 Ties 和 non-Ties 样本
        ties_samples = []
        non_ties_samples = []

        for sample in samples:
            if self._detect_ties_mode(sample):
                ties_samples.append(sample)
            else:
                non_ties_samples.append(sample)

        print(f"处理 {len(non_ties_samples)} 个非Ties样本，跳过 {len(ties_samples)} 个Ties样本")
        print(f"使用 {max_workers} 个并行工作线程")
        print(f"比较模式: {self.reward.comparison_mode}")

        # 只对非Ties样本进行冲突检测
        non_ties_results = []
        if non_ties_samples:
            print("开始冲突检测评估（仅非Ties样本）...")
            non_ties_results = self._parallel_evaluate(
                non_ties_samples, "冲突检测样本", max_workers, **kwargs
            )

        # 生成摘要
        try:
            summary = self.summary(non_ties_results)
            summary.update(
                {
                    "total_count": len(samples),
                    "non_ties_count": len(non_ties_samples),
                    "ties_count": len(ties_samples),
                    "max_workers": max_workers,
                    "comparison_mode": self.reward.comparison_mode,
                }
            )
            return summary
        except Exception as e:
            return {"error": f"摘要生成失败: {str(e)}"}

    def _calculate_conflict_metrics(self, results: List[DataSample]) -> ConflictMetrics:
        """计算冲突指标，基于新的强连通分量（SCC）检测方法"""
        if not results:
            return ConflictMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

        total_samples = 0
        total_scc_conflicts = 0  # 总强连通分量冲突数量
        total_comparisons = 0
        samples_with_conflicts = 0
        total_conflicts_detected = 0  # 总检测到的冲突实例

        for sample in results:
            try:
                # 跳过有评估错误的样本
                if sample.metadata and sample.metadata.get("conflict_evaluation_error"):
                    continue

                # 获取评估结果
                if (
                    not sample.metadata
                    or "conflict_evaluation_result" not in sample.metadata
                ):
                    continue

                eval_result = sample.metadata["conflict_evaluation_result"]

                # 从新的数据结构中获取冲突信息
                conflicts = eval_result.get("conflicts", [])
                comparison_results = eval_result.get("comparison_results", {})

                # 统计样本
                total_samples += 1

                # 统计比较次数
                sample_comparisons = len(comparison_results)
                total_comparisons += sample_comparisons

                # 新的冲突统计逻辑：基于强连通分量
                sample_has_conflicts = len(conflicts) > 0
                if sample_has_conflicts:
                    samples_with_conflicts += 1
                    total_conflicts_detected += len(conflicts)

                    # 统计强连通分量冲突的总规模
                    for conflict in conflicts:
                        if conflict.get("type") == ConflictType.CYCLE.value:
                            # severity字段现在表示环路大小
                            scc_size = int(conflict.get("severity", 0))
                            total_scc_conflicts += scc_size

            except Exception as e:
                logger.debug(f"处理样本时出错: {str(e)}")
                pass

        # 计算冲突率（百分比形式，越低越好）
        overall_conflict_rate = (
            (samples_with_conflicts / total_samples * 100) if total_samples > 0 else 0.0
        )

        # 对于SCC方法，传递性冲突和循环冲突率都等于总体冲突率
        # 因为所有冲突都通过强连通分量检测统一处理
        transitivity_conflict_rate = overall_conflict_rate
        cycle_conflict_rate = overall_conflict_rate

        # 计算冲突密度率（平均每个样本的SCC冲突节点数）
        conflict_density_rate = (
            (total_scc_conflicts / total_samples) if total_samples > 0 else 0.0
        )

        # 计算每次比较的平均冲突节点数
        conflicts_per_comparison = (
            (total_scc_conflicts / total_comparisons) if total_comparisons > 0 else 0.0
        )

        # 添加调试信息
        logger.debug(
            f"SCC冲突统计: 总样本={total_samples}, 总SCC冲突节点数={total_scc_conflicts}, 总比较次数={total_comparisons}"
        )
        logger.debug(
            f"有冲突样本={samples_with_conflicts}, 冲突检测实例={total_conflicts_detected}"
        )
        logger.debug(
            f"平均每样本SCC冲突节点数={conflict_density_rate:.2f}, 每比较SCC冲突节点数={conflicts_per_comparison:.4f}"
        )

        return ConflictMetrics(
            overall_conflict_rate=overall_conflict_rate,
            transitivity_conflict_rate=transitivity_conflict_rate,
            cycle_conflict_rate=cycle_conflict_rate,
            conflict_density_rate=conflict_density_rate,
            conflicts_per_comparison=conflicts_per_comparison,
            total_samples=total_samples,
            total_conflicts=total_scc_conflicts,  # 现在指SCC冲突节点总数
            total_comparisons=total_comparisons,
        )

    def _calculate_accuracy_metrics(
        self, results: List[DataSample]
    ) -> Dict[str, float]:
        """计算准确率指标（基于chosen vs rejected的比较结果）"""
        if not results:
            return {
                "accuracy": 0.0,
                "total_chosen_wins": 0,
                "total_chosen_losses": 0,
                "total_chosen_ties": 0,
                "total_chosen_vs_rejected_comparisons": 0,
                "chosen_dominance_rate": 0.0,
                "total_dominance_samples": 0,
                "valid_samples": 0,
                "total_samples": 0,
                "strategy": "chosen_vs_rejected_only",
            }

        total_chosen_wins = 0
        total_chosen_losses = 0
        total_chosen_ties = 0
        total_chosen_vs_rejected_comparisons = 0
        total_dominance_samples = 0
        valid_count = 0

        for sample in results:
            try:
                # 跳过有评估错误的样本
                if sample.metadata and sample.metadata.get("conflict_evaluation_error"):
                    continue

                # 获取评估结果
                if (
                    not sample.metadata
                    or "conflict_evaluation_result" not in sample.metadata
                ):
                    continue

                eval_result = sample.metadata["conflict_evaluation_result"]
                accuracy_data = eval_result.get("accuracy_data", {})

                # 跳过有错误的样本
                if "error" in accuracy_data:
                    continue

                # 累计chosen vs rejected的比较统计
                total_chosen_wins += accuracy_data.get("chosen_wins", 0)
                total_chosen_losses += accuracy_data.get("chosen_losses", 0)
                total_chosen_ties += accuracy_data.get("chosen_ties", 0)
                total_chosen_vs_rejected_comparisons += accuracy_data.get(
                    "chosen_vs_rejected_comparisons", 0
                )

                # 累计主导度统计
                total_dominance_samples += accuracy_data.get("chosen_dominance", 0)

                valid_count += 1

            except Exception as e:
                logger.debug(f"处理样本准确率时出错: {str(e)}")
                pass

        # 计算总体准确率：所有chosen vs rejected比较中，chosen获胜的比例
        accuracy = (
            total_chosen_wins / total_chosen_vs_rejected_comparisons
            if total_chosen_vs_rejected_comparisons > 0
            else 0.0
        )

        # 计算主导率：chosen独占胜利数最高的样本比例
        chosen_dominance_rate = (
            total_dominance_samples / valid_count if valid_count > 0 else 0.0
        )

        return {
            "accuracy": float(accuracy),
            "total_chosen_wins": total_chosen_wins,
            "total_chosen_losses": total_chosen_losses,
            "total_chosen_ties": total_chosen_ties,
            "total_chosen_vs_rejected_comparisons": total_chosen_vs_rejected_comparisons,
            "chosen_dominance_rate": float(chosen_dominance_rate),
            "total_dominance_samples": total_dominance_samples,
            "valid_samples": valid_count,
            "total_samples": len(results),
            "chosen_win_rate": float(accuracy),
            "chosen_loss_rate": total_chosen_losses
            / total_chosen_vs_rejected_comparisons
            if total_chosen_vs_rejected_comparisons > 0
            else 0.0,
            "chosen_tie_rate": total_chosen_ties / total_chosen_vs_rejected_comparisons
            if total_chosen_vs_rejected_comparisons > 0
            else 0.0,
            "avg_comparisons_per_sample": total_chosen_vs_rejected_comparisons
            / valid_count
            if valid_count > 0
            else 0.0,
            "strategy": "chosen_vs_rejected_only",
        }

    def summary(self, results: List[DataSample]) -> dict:
        """生成评估摘要"""
        # 计算冲突指标
        conflict_metrics = self._calculate_conflict_metrics(results)

        # 计算准确率指标
        accuracy_metrics = self._calculate_accuracy_metrics(results)

        # 计算基础统计
        successful_samples = 0
        failed_samples = 0
        total_model_calls = 0
        total_expected_comparisons = 0
        total_successful_comparisons = 0

        for sample in results:
            if sample.metadata and sample.metadata.get("conflict_evaluation_error"):
                failed_samples += 1
            elif sample.metadata and "conflict_evaluation_result" in sample.metadata:
                successful_samples += 1

                # 统计模型调用次数
                extra_data = sample.metadata["conflict_evaluation_result"].get(
                    "extra_data", {}
                )
                comparison_quality = extra_data.get("comparison_quality", {})

                expected = comparison_quality.get("expected_comparisons", 0)
                successful = comparison_quality.get("successful_comparisons", 0)

                total_expected_comparisons += expected
                total_successful_comparisons += successful

                # 根据比较模式统计模型调用次数
                if self.reward.comparison_mode == "pairwise":
                    total_model_calls += successful  # pairwise: 每个比较对1次调用
                else:  # pointwise
                    total_model_calls += successful * 2  # pointwise: 每个比较对2次调用

        # 构建摘要
        return {
            "model": self.reward.llm.model if self.reward.llm else "unknown",
            "comparison_mode": self.reward.comparison_mode,
            "accuracy_metrics": accuracy_metrics,
            "conflict_metrics": {
                "overall_conflict_rate": conflict_metrics.overall_conflict_rate,
                "transitivity_conflict_rate": conflict_metrics.transitivity_conflict_rate,
                "cycle_conflict_rate": conflict_metrics.cycle_conflict_rate,
                "conflict_density_rate": conflict_metrics.conflict_density_rate,
                "conflicts_per_comparison": conflict_metrics.conflicts_per_comparison,
                "total_samples": conflict_metrics.total_samples,
                "total_conflicts": conflict_metrics.total_conflicts,
                "total_comparisons": conflict_metrics.total_comparisons,
            },
            "evaluation_summary": {
                "successful_samples": successful_samples,
                "failed_samples": failed_samples,
                "success_rate": successful_samples / len(results) if results else 0,
                "total_model_calls": total_model_calls,
                "total_expected_comparisons": total_expected_comparisons,
                "total_successful_comparisons": total_successful_comparisons,
                "comparison_success_rate": total_successful_comparisons
                / total_expected_comparisons
                if total_expected_comparisons > 0
                else 0,
            },
        }


def main(
    data_path: str = "data/benchmarks/reward-bench-2/data/test-00000-of-00002.parquet",
    result_path: str = "data/results/conflict_detection.json",
    max_samples: int = -1,
    model: str | dict = "qwen2.5-72b-instruct",
    max_workers: int = 8,
    comparison_mode: str = "pairwise",
    save_detailed_outputs: bool = True,
    random_seed: int = 42,
):
    """冲突检测主评估管道，采用RewardBench2风格实现

    支持两种比较模式：pairwise（直接两两比较）和 pointwise（独立评分后比较）

    Args:
        data_path: 输入数据集文件路径
        result_path: 保存评估结果的路径
        max_samples: 处理的最大样本数（-1表示全部）
        model: 模型标识符字符串或配置字典
        max_workers: 并行评估的最大工作线程数
        comparison_mode: 比较模式，"pairwise"或"pointwise"
        save_detailed_outputs: 是否保存详细的模型输出记录
        random_seed: 随机数种子，确保抽样的可重复性（仅在max_samples != -1时生效）
    """
    try:
        # 验证输入参数
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件未找到: {data_path}")

        if comparison_mode not in ["pairwise", "pointwise"]:
            raise ValueError(f"不支持的比较模式: {comparison_mode}，支持的模式: pairwise, pointwise")

        # 设置随机数种子以确保可重复性
        if max_samples > 0:
            # 只在需要抽样时设置随机种子
            random.seed(random_seed)
            np.random.seed(random_seed)
            print(f"设置随机数种子: {random_seed} (用于样本抽取)")

        if max_samples <= 0:
            max_samples = None  # 加载所有样本

        # 创建数据加载配置
        config = {
            "path": data_path,
            "limit": max_samples,
        }

        # 初始化数据加载模块
        print(f"从以下位置加载数据: {data_path}")
        load_module = create_loader(
            name="rewardbench2",
            load_strategy_type="local",
            data_source="rewardbench2",
            config=config,
        )

        # 初始化语言模型用于评估
        print(f"初始化模型: {model}")
        if isinstance(model, str):
            llm = OpenaiLLM(model=model)
        elif isinstance(model, dict):
            llm = OpenaiLLM(**model)
        else:
            raise ValueError(f"无效的模型类型: {type(model)}。期望str或dict类型。")

        # 加载评估数据集
        dataset = load_module.run()
        samples = dataset.get_data_samples()
        print(f"加载了 {len(samples)} 个样本用于评估")

        if not samples:
            print("未加载到样本。请检查数据文件和配置。")
            return

        # 创建评估器实例
        evaluator = ConflictDetectionEvaluator(
            reward=ConflictDetectionReward(
                name="conflict_detection",
                llm=llm,
                comparison_mode=comparison_mode,
                save_detailed_outputs=save_detailed_outputs,
            )
        )

        # 执行评估管道，支持并行处理
        results = evaluator.run(samples=samples, max_workers=max_workers)

        # 打印详细评估结果
        print_evaluation_results(results)

        # 确保结果目录存在
        result_dir = os.path.dirname(result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        # 将评估结果持久化到文件
        print(f"结果保存到: {result_path}")
        write_json(results, result_path)

        print("评估完成成功！")

    except Exception as e:
        print(f"评估失败: {e}")
        raise


def print_evaluation_results(results: dict):
    """打印详细的评估结果，与rewardbench2.py风格保持一致"""
    print("\n" + "=" * 80)
    print("冲突检测评估结果")
    print("=" * 80)

    print(f"\n模型: {results.get('model', 'Unknown')}")
    print(f"比较模式: {results.get('comparison_mode', 'Unknown')}")

    # 打印准确率指标
    accuracy_metrics = results.get("accuracy_metrics", {})
    if accuracy_metrics:
        strategy = accuracy_metrics.get("strategy", "unknown")
        print("\n准确率指标 (基于chosen vs rejected的比较):")
        print(
            f"  准确率: {accuracy_metrics.get('accuracy', 0):.4f} ({accuracy_metrics.get('accuracy', 0)*100:.2f}%)"
        )
        print(
            f"  Chosen主导率: {accuracy_metrics.get('chosen_dominance_rate', 0):.4f} ({accuracy_metrics.get('chosen_dominance_rate', 0)*100:.2f}%)"
        )
        print(
            f"  Chosen获胜: {accuracy_metrics.get('total_chosen_wins', 0)}/{accuracy_metrics.get('total_chosen_vs_rejected_comparisons', 0)}"
        )
        print(
            f"  Chosen失败: {accuracy_metrics.get('total_chosen_losses', 0)} ({accuracy_metrics.get('chosen_loss_rate', 0)*100:.1f}%)"
        )
        print(
            f"  平局: {accuracy_metrics.get('total_chosen_ties', 0)} ({accuracy_metrics.get('chosen_tie_rate', 0)*100:.1f}%)"
        )
        print(
            f"  主导样本数: {accuracy_metrics.get('total_dominance_samples', 0)}/{accuracy_metrics.get('valid_samples', 0)}"
        )
        print(
            f"  有效样本数: {accuracy_metrics.get('valid_samples', 0)}/{accuracy_metrics.get('total_samples', 0)}"
        )
        print(
            f"  平均每样本比较次数: {accuracy_metrics.get('avg_comparisons_per_sample', 0):.1f}"
        )
        print(f"  评估策略: {strategy}")
        print("  注: 主导率指chosen独占胜利数最高的样本比例")

    # 打印冲突指标
    conflict_metrics = results.get("conflict_metrics", {})
    if conflict_metrics:
        print("\n核心冲突指标 (基于强连通分量检测，越低越好):")
        print(f"  总体冲突率: {conflict_metrics.get('overall_conflict_rate', 0):.2f}%")
        print(f"  平均每样本冲突节点数: {conflict_metrics.get('conflict_density_rate', 0):.2f}")
        print(
            f"  平均每比较冲突节点数: {conflict_metrics.get('conflicts_per_comparison', 0):.4f}"
        )
        print("  检测方法: Tarjan强连通分量算法")

        print("\n详细统计:")
        print(f"  总样本数: {conflict_metrics.get('total_samples', 0)}")
        print(f"  总SCC冲突节点数: {conflict_metrics.get('total_conflicts', 0)}")
        print(f"  总比较次数: {conflict_metrics.get('total_comparisons', 0)}")

    # 打印评估摘要
    eval_summary = results.get("evaluation_summary", {})
    if eval_summary:
        print("\n评估摘要:")
        print(f"  成功评估样本: {eval_summary.get('successful_samples', 0)}")
        print(f"  失败样本: {eval_summary.get('failed_samples', 0)}")
        print(f"  成功率: {eval_summary.get('success_rate', 0):.2%}")
        print(f"  总模型调用次数: {eval_summary.get('total_model_calls', 0)}")
        print(f"  预期比较次数: {eval_summary.get('total_expected_comparisons', 0)}")
        print(f"  成功比较次数: {eval_summary.get('total_successful_comparisons', 0)}")
        print(f"  比较成功率: {eval_summary.get('comparison_success_rate', 0):.2%}")

    # 打印总体统计
    print("\n总体统计:")
    print(f"  处理的样本总数: {results.get('total_count', 0)}")
    print(f"  非Ties样本数: {results.get('non_ties_count', 0)}")
    print(f"  Ties样本数（已跳过）: {results.get('ties_count', 0)}")
    print(f"  使用的工作线程数: {results.get('max_workers', 0)}")

    # 性能解释
    comparison_mode = results.get("comparison_mode", "unknown")
    if comparison_mode == "pairwise":
        print("  比较策略: 直接两两比较 (每个比较对1次模型调用)")
    elif comparison_mode == "pointwise":
        print("  比较策略: 独立评分后比较 (每个比较对2次模型调用)")

    # 冲突指标解释
    if conflict_metrics.get("overall_conflict_rate", 0) > 0:
        print("\n📊 冲突分析:")
        overall_rate = conflict_metrics.get("overall_conflict_rate", 0)
        if overall_rate < 5:
            assessment = "极低 - 模型表现优秀"
        elif overall_rate < 15:
            assessment = "较低 - 模型表现良好"
        elif overall_rate < 30:
            assessment = "中等 - 模型有一定改进空间"
        else:
            assessment = "较高 - 模型存在逻辑一致性问题"
        print(f"  冲突率评估: {assessment}")

        avg_conflicts = conflict_metrics.get("conflict_density_rate", 0)
        conflicts_per_comp = conflict_metrics.get("conflicts_per_comparison", 0)
        if avg_conflicts > 0:
            print(f"  冲突密度: 平均每个样本{avg_conflicts:.2f}个冲突节点")
            if conflicts_per_comp > 0:
                print(f"  冲突强度: 平均每次比较涉及{conflicts_per_comp:.4f}个冲突节点")
        print("  技术说明: 强连通分量大小>1表示存在环路，环路中的节点数即为冲突节点数")
    else:
        print("\n✅ 无冲突检测 - 模型在该数据集上表现出完美的逻辑一致性（无环路）！")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
