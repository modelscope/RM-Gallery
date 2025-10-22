"""
Conflict Detector
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Dict, List, Optional

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
        PAIRWISE_CYCLE: Two-node cycle (A>B and B>A)
        MULTI_CYCLE: Multi-node cycle (A>B>C>...>A, at least 3 nodes)
    """

    PAIRWISE_CYCLE = "pairwise_cycle"  # Two-node cycle conflict (A>B and B>A)
    MULTI_CYCLE = "multi_cycle"  # Multi-node cycle conflict (≥3 nodes)


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
    """Comparison result enumeration"""

    A_BETTER = 1  # A > B
    B_BETTER = -1  # A < B
    TIE = 0  # A = B


@dataclass
class ConflictMetrics:
    """Conflict metrics based on SCC detection (percentage form, lower is better)"""

    overall_conflict_rate: float  # Overall conflict rate (%) - percentage of samples with conflicts

    # Detailed statistics
    total_samples: int
    samples_with_conflicts: int  # Number of samples with conflicts
    total_comparisons: int  # Total successful comparisons


class ConflictDetector:
    """Core conflict detector class, fully ported from original algorithm"""

    def __init__(self):
        pass

    def build_comparison_matrix(
        self, responses: List[str], comparison_results: Dict[tuple, ComparisonResult]
    ) -> np.ndarray:
        """
        Build comparison matrix

        Args:
            responses: List of responses
            comparison_results: Comparison result dictionary {(i,j): ComparisonResult}

        Returns:
            Comparison matrix M[i][j] = 1 means i>j, -1 means i<j, 0 means i=j
        """
        n = len(responses)
        matrix = np.zeros((n, n), dtype=int)

        for (i, j), result in comparison_results.items():
            matrix[i][j] = result.value
            matrix[j][i] = -result.value  # Symmetric fill

        return matrix

    def detect_symmetry_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """Detect symmetry conflicts (no such conflicts in current single-comparison scenario)"""
        # In our implementation, each pair of responses is compared only once, then automatically fills symmetric positions
        # Therefore, there will be no real symmetry conflicts, return empty list
        return []

    def detect_transitivity_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """Detect transitivity conflicts"""
        conflicts = []
        n = matrix.shape[0]

        # Add debug information
        total_checks = 0
        valid_chains = 0

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        total_checks += 1
                        # Check transitivity: if i>j and j>k, then i>k should hold
                        if matrix[i][j] > 0 and matrix[j][k] > 0:
                            valid_chains += 1
                            if matrix[i][k] <= 0:
                                conflicts.append(
                                    Conflict(
                                        conflict_type=ConflictType.TRANSITIVITY,
                                        involved_items=[i, j, k],
                                        description=f"Transitivity conflict: response{i}>response{j}>response{k}, but response{i}{'<' if matrix[i][k] < 0 else '='}response{k}",
                                    )
                                )

        # Output debug information only when there is a comparison matrix
        if total_checks > 0:
            logger.debug(
                f"Transitivity check: total_checks={total_checks}, valid_chains={valid_chains}, conflicts={len(conflicts)}"
            )
            logger.debug(f"Comparison matrix:\n{matrix}")

        return conflicts

    def detect_cycles(self, matrix: np.ndarray) -> List[Conflict]:
        """Detect cycle conflicts using DFS"""
        conflicts = []
        n = matrix.shape[0]

        def dfs_cycle_detection(node: int, path: List[int], visited: set) -> bool:
            """DFS cycle detection"""
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle_nodes = path[cycle_start:] + [node]
                if len(cycle_nodes) > 2:  # Cycle with at least 3 nodes
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.CYCLE,
                            involved_items=cycle_nodes[
                                :-1
                            ],  # Remove duplicate last node
                            description=f"Cycle conflict: response {' > '.join(map(str, cycle_nodes))} forms a cycle",
                            severity=len(cycle_nodes) - 1,
                        )
                    )
                return True

            if node in visited:
                return False

            visited.add(node)
            path.append(node)

            # Visit all child nodes (j where i>j and matrix[i][j]>0)
            for next_node in range(n):
                if matrix[node][next_node] > 0:
                    if dfs_cycle_detection(next_node, path, visited):
                        return True

            path.pop()
            return False

        # Start cycle detection from each node
        for start_node in range(n):
            visited = set()
            dfs_cycle_detection(start_node, [], visited)

        # Deduplicate (same cycle might be detected multiple times)
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
        Detect conflicts using Strongly Connected Components (SCC)
        Based on definition: existence of SCC with size > 1 indicates cycle conflicts

        Args:
            matrix: Comparison matrix M[i][j] = 1 means i>j, -1 means i<j, 0 means i=j

        Returns:
            True if conflicts exist, False otherwise
        """
        n = matrix.shape[0]
        if n < 2:
            return False  # Less than 2 nodes cannot form a cycle

        # Build directed graph adjacency list
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i][j] > 0:
                    graph[i].append(j)  # i -> j

        # Use Tarjan's algorithm to detect strongly connected components
        sccs = self._tarjan_scc(graph)

        # Check if there exists SCC with size > 1
        for scc in sccs:
            if len(scc) > 1:
                return True

        return False

    def _tarjan_scc(self, graph: List[List[int]]) -> List[List[int]]:
        """
        Tarjan's algorithm implementation for strongly connected components detection

        Args:
            graph: Directed graph represented as adjacency list

        Returns:
            List of strongly connected components, each component is a list of node indices
        """
        n = len(graph)
        index = 0
        stack = []
        indices = [-1] * n  # Visit order of nodes
        lowlinks = [-1] * n  # Earliest ancestor that a node can backtrack to
        on_stack = [False] * n
        sccs = []

        def strongconnect(v):
            nonlocal index
            # Set depth index of node v
            indices[v] = index
            lowlinks[v] = index
            index += 1
            stack.append(v)
            on_stack[v] = True

            # Traverse all successor nodes of v
            for w in graph[v]:
                if indices[w] == -1:
                    # Successor node w not yet visited, recursively visit
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack[w]:
                    # Successor node w is on stack, indicating a cycle is found
                    lowlinks[v] = min(lowlinks[v], indices[w])

            # If v is the root node of a strongly connected component
            if lowlinks[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        # Execute DFS on all unvisited nodes
        for v in range(n):
            if indices[v] == -1:
                strongconnect(v)

        return sccs

    def count_scc_conflicts(self, matrix: np.ndarray) -> int:
        """
        Count the number of strongly connected components with cycle conflicts

        Args:
            matrix: Comparison matrix

        Returns:
            Number of SCCs with size > 1 (i.e., number of cycle conflicts)
        """
        n = matrix.shape[0]
        if n < 2:
            return 0

        # Build directed graph adjacency list
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i][j] > 0:
                    graph[i].append(j)  # i -> j

        # Use Tarjan's algorithm to detect strongly connected components
        sccs = self._tarjan_scc(graph)

        # Count SCCs with size > 1
        conflict_count = sum(1 for scc in sccs if len(scc) > 1)

        return conflict_count

    def detect_all_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """
        Detect all types of conflicts
        Using SCC detection method, distinguishing between two-node and multi-node cycles
        """
        conflicts = []

        # Use new SCC method to detect conflicts
        if self.has_conflicts_scc_detection(matrix):
            # If conflicts exist, get specific strongly connected component information
            n = matrix.shape[0]
            graph = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j and matrix[i][j] > 0:
                        graph[i].append(j)

            sccs = self._tarjan_scc(graph)
            conflict_sccs = [scc for scc in sccs if len(scc) > 1]

            # Create a conflict record for each conflicting strongly connected component
            for idx, scc in enumerate(conflict_sccs):
                if len(scc) == 2:
                    # Two-node cycle (mutual preference conflict)
                    conflict_type = ConflictType.PAIRWISE_CYCLE
                    description = f"Bidirectional conflict: mutual preference exists between response{scc[0]} and response{scc[1]}"
                else:
                    # Multi-node cycle
                    conflict_type = ConflictType.MULTI_CYCLE
                    description = f"Cycle conflict: response {' -> '.join(map(str, scc + [scc[0]]))} forms a {len(scc)}-node cycle"

                conflicts.append(
                    Conflict(
                        conflict_type=conflict_type,
                        involved_items=scc,
                        description=description,
                        severity=float(len(scc)),  # Cycle size as severity
                    )
                )

        return conflicts


def extract_rewardbench2_responses(sample: DataSample) -> List[str]:
    """Extract responses from RewardBench2 sample (1 chosen + 3 rejected)"""
    responses = []

    # Extract all response content from output
    for output in sample.output:
        responses.append(output.answer.content)

    # Ensure we have at least 2 responses for comparison
    if len(responses) < 2:
        # If insufficient responses, fill with duplicate responses
        while len(responses) < 2:
            responses.append(responses[0] if responses else "No response")

    return responses


def extract_prompt(sample: DataSample) -> str:
    """Extract prompt from sample"""
    if sample.input and len(sample.input) > 0:
        return sample.input[-1].content
    return ""


def create_comparison_pairs(responses: List[str]) -> List[tuple]:
    """Generate indices for all pairwise comparisons"""
    pairs = []
    n = len(responses)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


class ConflictDetectionReward(BaseLLMReward, BasePairWiseReward):
    """Reward model supporting conflict detection, supports both pairwise and pointwise comparison modes"""

    model_config = {"arbitrary_types_allowed": True}

    comparison_mode: str = Field(
        default="pairwise", description="Comparison mode: 'pairwise' or 'pointwise'"
    )
    pointwise_template: PointwiseTemplate = Field(
        default_factory=PointwiseTemplate, description="Pointwise scoring template"
    )
    pairwise_template: PairComparisonTemplate = Field(
        default_factory=PairComparisonTemplate,
        description="Pairwise comparison template",
    )
    conflict_detector: ConflictDetector = Field(
        default_factory=ConflictDetector, description="Conflict detector instance"
    )
    save_detailed_outputs: bool = Field(
        default=True, description="Whether to save detailed model output records"
    )

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """Evaluate sample and detect conflicts"""
        assert self.llm is not None

        try:
            if self.comparison_mode == "pointwise":
                return self._evaluate_pointwise_mode(sample, **kwargs)
            else:
                return self._evaluate_pairwise_mode(sample, **kwargs)
        except Exception as e:
            logger.error(f"Sample evaluation failed: {str(e)}")
            return RewardResult(
                name=self.name, details=[], extra_data={"error": str(e)}
            )

    def _evaluate_pairwise_mode(self, sample: DataSample, **kwargs) -> RewardResult:
        """Pairwise mode: direct pairwise comparison"""
        prompt = extract_prompt(sample)
        responses = extract_rewardbench2_responses(sample)

        if len(responses) < 2:
            logger.warning("Insufficient responses, skipping evaluation")
            return RewardResult(
                name=self.name,
                details=[],
                extra_data={"error": "insufficient_responses"},
            )

        # Get all comparison pairs
        comparison_pairs = create_comparison_pairs(responses)
        comparison_results = {}
        detailed_comparisons = []

        # Execute all pairwise comparisons
        for i, j in comparison_pairs:
            try:
                # Use pairwise template for comparison
                comparison_prompt = self.pairwise_template.format(
                    query=prompt, answers=[responses[i], responses[j]]
                )

                # Call LLM
                response_text = self.llm.simple_chat(query=comparison_prompt)

                # Parse result
                parsed = self.pairwise_template.parse(response_text)

                # Convert to ComparisonResult
                if parsed.best_answer.lower() == "a":
                    result = ComparisonResult.A_BETTER
                elif parsed.best_answer.lower() == "b":
                    result = ComparisonResult.B_BETTER
                else:
                    result = ComparisonResult.TIE

                comparison_results[(i, j)] = result

                # Save detailed information
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
                logger.warning(f"Comparison failed ({i}, {j}): {e}")
                comparison_results[(i, j)] = ComparisonResult.TIE

        return self._process_comparisons_and_detect_conflicts(
            responses, comparison_results, detailed_comparisons
        )

    def _evaluate_pointwise_mode(self, sample: DataSample, **kwargs) -> RewardResult:
        """Pointwise mode: independent scoring then comparison"""
        prompt = extract_prompt(sample)
        responses = extract_rewardbench2_responses(sample)

        if len(responses) < 2:
            logger.warning("Insufficient responses, skipping evaluation")
            return RewardResult(
                name=self.name,
                details=[],
                extra_data={"error": "insufficient_responses"},
            )

        # Get all comparison pairs
        comparison_pairs = create_comparison_pairs(responses)
        comparison_results = {}
        detailed_comparisons = []

        # For each comparison pair, score both responses in parallel
        for i, j in comparison_pairs:
            try:
                # Score both responses in parallel for efficiency
                score_i, score_j = self._score_pair_parallel(
                    prompt, responses[i], responses[j]
                )

                # Check if scoring succeeded
                if score_i is not None and score_j is not None:
                    # Compare scores to determine partial order
                    if score_i > score_j:
                        result = ComparisonResult.A_BETTER
                    elif score_j > score_i:
                        result = ComparisonResult.B_BETTER
                    else:
                        result = ComparisonResult.TIE

                    comparison_results[(i, j)] = result

                    # Save detailed information
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
                    # Scoring failed, skip this comparison pair
                    logger.warning(
                        f"Pointwise scoring failed, skipping comparison pair ({i}, {j}): Score A: {score_i}, Score B: {score_j}"
                    )

            except Exception as e:
                logger.warning(f"Pointwise comparison failed ({i}, {j}): {e}")

        return self._process_comparisons_and_detect_conflicts(
            responses, comparison_results, detailed_comparisons
        )

    def _score_single_response(self, prompt: str, response: str) -> Optional[float]:
        """
        Score a single response using pointwise template

        Args:
            prompt: Original question
            response: Response to score

        Returns:
            Score (1-10), returns None on failure
        """
        try:
            # Use pointwise template to generate scoring prompt
            scoring_prompt = self.pointwise_template.format(
                query=prompt, response=response
            )

            # Call model for scoring
            model_response = self.llm.simple_chat(query=scoring_prompt)

            # Parse scoring result
            parsed = self.pointwise_template.parse(model_response)

            # Check if score is in valid range
            if 1.0 <= parsed.score <= 10.0:
                return parsed.score
            else:
                logger.warning(f"Score out of range: {parsed.score}")
                return None

        except Exception as e:
            logger.warning(f"Scoring failed: {e}")
            return None

    def _score_pair_parallel(
        self, prompt: str, response_a: str, response_b: str
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Score a pair of responses in parallel for efficiency

        Args:
            prompt: Original question
            response_a: First response
            response_b: Second response

        Returns:
            (score_a, score_b) tuple, None at corresponding position on failure
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both scoring tasks simultaneously
            future_a = executor.submit(self._score_single_response, prompt, response_a)
            future_b = executor.submit(self._score_single_response, prompt, response_b)

            # Wait for both tasks to complete
            score_a = future_a.result()
            score_b = future_b.result()

            return score_a, score_b

    def _process_comparisons_and_detect_conflicts(
        self,
        responses: List[str],
        comparison_results: Dict[tuple, ComparisonResult],
        detailed_comparisons: List[Dict],
    ) -> RewardResult:
        """Process comparison results and detect conflicts"""

        # Build comparison matrix
        matrix = self.conflict_detector.build_comparison_matrix(
            responses, comparison_results
        )

        # Detect conflicts
        conflicts = self.conflict_detector.detect_all_conflicts(matrix)

        # Calculate statistics
        expected_comparisons = len(create_comparison_pairs(responses))
        successful_comparisons = len(comparison_results)

        # Calculate conflict statistics
        conflict_types = {ct.value: 0 for ct in ConflictType}
        for conflict in conflicts:
            conflict_types[conflict.conflict_type.value] += 1

        # Build extra_data
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

        # Add detailed output
        if self.save_detailed_outputs and detailed_comparisons:
            extra_data["detailed_comparisons"] = detailed_comparisons

        # Create simple rank scores (for compatibility)
        scores = [0.0] * len(responses)
        if len(conflicts) == 0:
            # When no conflicts, give first response higher score
            scores[0] = 1.0

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=f"{self.name}_conflict_detection",
                    reason=f"Detected {len(conflicts)} conflicts, successfully compared {successful_comparisons}/{expected_comparisons}",
                    rank=scores,
                )
            ],
            extra_data=extra_data,
        )

    def _parallel(self, func, sample, thread_pool=None, **kwargs) -> DataSample:
        """Streamlined parallel processing method, supports both pairwise and pointwise modes"""
        sample = sample.model_copy(deep=True)

        # Extract data
        prompt = extract_prompt(sample)
        responses = extract_rewardbench2_responses(sample)

        if len(responses) < 2:
            sample.input[-1].additional_kwargs = {
                "conflict_detector": {"comparison_results": {}}
            }
            return sample

        # Generate comparison pairs
        pairs = create_comparison_pairs(responses)
        comparison_results = {}

        # Perform comparisons based on mode
        for i, j in pairs:
            try:
                if self.comparison_mode == "pointwise":
                    # pointwise: parallel scoring then comparison for efficiency
                    score_a, score_b = self._score_pair_parallel(
                        prompt, responses[i], responses[j]
                    )
                    # Handle scoring failure, use default score 5.0
                    if score_a is None:
                        score_a = 5.0
                    if score_b is None:
                        score_b = 5.0
                    comparison_results[(i, j)] = (
                        1 if score_a > score_b else (-1 if score_b > score_a else 0)
                    )
                else:
                    # pairwise: direct comparison
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
                logger.warning(f"Comparison failed ({i}, {j}): {e}")
                comparison_results[(i, j)] = 0

        # Store results
        sample.input[-1].additional_kwargs = {
            "conflict_detector": {"comparison_results": comparison_results}
        }
        return sample

    def _score_response(self, prompt: str, response: str) -> float:
        """Score a single response for pointwise mode"""
        try:
            scoring_prompt = self.pointwise_template.format(
                query=prompt, response=response
            )
            model_response = self.llm.simple_chat(scoring_prompt)
            parsed_result = self.pointwise_template.parse(model_response)
            return parsed_result.score
        except Exception as e:
            logger.warning(f"Scoring failed: {e}")
            return 5.0


class ConflictDetectionEvaluator(BaseEvaluator):
    """Conflict detection evaluator, supports multi-threaded processing and conflict metrics calculation"""

    reward: ConflictDetectionReward = Field(
        default=..., description="Conflict detection reward model"
    )

    def _detect_ties_mode(self, sample: DataSample) -> bool:
        """Detect if sample is from Ties subset"""
        if hasattr(sample, "metadata") and sample.metadata:
            subset = sample.metadata.get("subset", "").lower()
            return subset == "ties"
        return False

    def _calculate_accuracy_for_sample(
        self, sample: DataSample, comparison_results: Dict[tuple, int]
    ) -> Dict:
        """Calculate accuracy for a single sample: only considers chosen vs rejected comparison results"""
        responses = extract_rewardbench2_responses(sample)
        n = len(responses)

        # Find indices of chosen and rejected responses
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

        # Count chosen vs rejected comparison results
        chosen_wins = 0
        chosen_losses = 0
        chosen_ties = 0
        total_chosen_vs_rejected = 0

        for (i, j), result in comparison_results.items():
            # Check if this is a chosen vs rejected comparison
            is_chosen_vs_rejected = False
            chosen_better = False

            if i == chosen_index and j in rejected_indices:
                is_chosen_vs_rejected = True
                chosen_better = result > 0  # chosen > rejected
            elif j == chosen_index and i in rejected_indices:
                is_chosen_vs_rejected = True
                chosen_better = (
                    result < 0
                )  # rejected < chosen (i.e., chosen > rejected)

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

        # Calculate accuracy: proportion of chosen wins
        accuracy = (
            chosen_wins / total_chosen_vs_rejected
            if total_chosen_vs_rejected > 0
            else 0.0
        )

        # Calculate chosen dominance: check if chosen has uniquely highest win count
        chosen_dominance = 0
        if comparison_results:  # Only calculate when there are comparison results
            # Count total wins for each response
            win_counts = [0] * n
            for (i, j), result in comparison_results.items():
                if result > 0:  # i > j
                    win_counts[i] += 1
                elif result < 0:  # i < j
                    win_counts[j] += 1

            # Check if chosen has uniquely highest win count
            if chosen_index is not None:
                chosen_win_count = win_counts[chosen_index]
                max_win_count = max(win_counts)

                # Chosen has uniquely highest win count
                if (
                    chosen_win_count == max_win_count
                    and win_counts.count(max_win_count) == 1
                ):
                    chosen_dominance = 1

        return {
            "is_correct": accuracy
            > 0.5,  # Correct if chosen wins more than half of comparisons
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
        """Evaluate a single sample - for parallel processing"""
        try:
            # Randomly shuffle response order to avoid position bias
            # Use hash of sample content as seed to ensure same shuffling across different models
            import hashlib
            import random

            sample_copy = sample.model_copy(deep=True)

            # Create deterministic seed based on sample content
            sample_hash = hashlib.md5(str(sample.input).encode()).hexdigest()
            sample_seed = int(sample_hash[:8], 16)  # Use first 8 chars of hash as seed

            # Temporarily set random seed for shuffling
            random_state = random.getstate()  # Save current random state
            random.seed(sample_seed)
            random.shuffle(sample_copy.output)
            random.setstate(random_state)  # Restore previous random state

            # Use _parallel method to process all comparison pairs in sample
            processed_sample = self.reward._parallel(
                func=self.reward._evaluate,
                sample=sample_copy,
                thread_pool=None,  # Let _parallel create its own thread pool
                **kwargs,
            )

            # Analyze conflicts - inline processing
            if "conflict_detector" in processed_sample.input[-1].additional_kwargs:
                conflict_data = processed_sample.input[-1].additional_kwargs[
                    "conflict_detector"
                ]
                comparison_results = conflict_data.get("comparison_results", {})

                if comparison_results:
                    responses = extract_rewardbench2_responses(sample_copy)
                    n = len(responses)

                    # Build comparison matrix
                    comparison_matrix = np.zeros((n, n), dtype=int)
                    for (i, j), score in comparison_results.items():
                        comparison_matrix[i][j] = score
                        comparison_matrix[j][i] = -score

                    # Detect conflicts
                    conflicts = self.reward.conflict_detector.detect_all_conflicts(
                        comparison_matrix
                    )
                    conflict_types = {
                        ct.value: sum(1 for c in conflicts if c.conflict_type == ct)
                        for ct in ConflictType
                    }

                    # Update conflict data
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

            # Calculate accuracy: whether the one with most wins is chosen (note: using shuffled sample_copy)
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

            # Store processing results in sample metadata
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
            logger.error(f"Sample evaluation failed: {str(e)}")
            # Return sample with error information
            sample.metadata = sample.metadata or {}
            sample.metadata["conflict_evaluation_error"] = str(e)
            return sample

    def _parallel_evaluate(
        self, samples: List[DataSample], desc: str, max_workers: int = 8, **kwargs
    ) -> List[DataSample]:
        """Parallel evaluation, consistent with rewardbench2.py style"""
        results = [None] * len(samples)
        completed_count = 0

        def update_progress_bar(done, total):
            """Simple progress indicator"""
            progress = int(50 * done / total) if total > 0 else 0
            print(
                f"\r[{'#' * progress}{'.' * (50 - progress)}] {done}/{total}",
                end="",
                flush=True,
            )

        # Create evaluation function with kwargs
        eval_func = partial(self._evaluate_single_sample, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and map to original indices
            future_to_index = {
                executor.submit(eval_func, sample): i
                for i, sample in enumerate(samples)
            }

            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=60)  # Add a timeout of 60 seconds
                    results[index] = result
                except Exception as e:
                    logger.error(f"Task failed, sample index {index}: {str(e)}")
                    # Create error result
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
        """Execute evaluation with parallel processing support"""
        if not samples:
            return {"error": "No samples to evaluate"}

        # Separate Ties and non-Ties samples
        ties_samples = []
        non_ties_samples = []

        for sample in samples:
            if self._detect_ties_mode(sample):
                ties_samples.append(sample)
            else:
                non_ties_samples.append(sample)

        print(
            f"Processing {len(non_ties_samples)} non-Ties samples, skipping {len(ties_samples)} Ties samples"
        )
        print(f"Using {max_workers} parallel worker threads")
        print(f"Comparison mode: {self.reward.comparison_mode}")

        # Only perform conflict detection on non-Ties samples
        non_ties_results = []
        if non_ties_samples:
            print("Starting conflict detection evaluation (non-Ties samples only)...")
            non_ties_results = self._parallel_evaluate(
                non_ties_samples, "Conflict detection samples", max_workers, **kwargs
            )

        # Generate summary
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
            return {"error": f"Summary generation failed: {str(e)}"}

    def _calculate_conflict_metrics(self, results: List[DataSample]) -> ConflictMetrics:
        """Calculate conflict metrics based on SCC detection method"""
        if not results:
            return ConflictMetrics(0.0, 0, 0, 0)

        total_samples = 0
        total_comparisons = 0
        samples_with_conflicts = 0

        for sample in results:
            try:
                # Skip samples with evaluation errors
                if sample.metadata and sample.metadata.get("conflict_evaluation_error"):
                    continue

                # Get evaluation result
                if (
                    not sample.metadata
                    or "conflict_evaluation_result" not in sample.metadata
                ):
                    continue

                eval_result = sample.metadata["conflict_evaluation_result"]

                # Get conflict information
                conflicts = eval_result.get("conflicts", [])
                comparison_results = eval_result.get("comparison_results", {})

                # Count samples
                total_samples += 1

                # Count comparisons
                sample_comparisons = len(comparison_results)
                total_comparisons += sample_comparisons

                # Count conflicts
                if len(conflicts) > 0:
                    samples_with_conflicts += 1

            except Exception as e:
                logger.debug(f"Error processing sample: {str(e)}")
                pass

        # Calculate conflict rate (percentage form, lower is better)
        overall_conflict_rate = (
            (samples_with_conflicts / total_samples * 100) if total_samples > 0 else 0.0
        )

        # Add debug information
        logger.debug(
            f"Conflict statistics: total_samples={total_samples}, samples_with_conflicts={samples_with_conflicts}, total_comparisons={total_comparisons}"
        )
        logger.debug(f"Overall conflict rate: {overall_conflict_rate:.2f}%")

        return ConflictMetrics(
            overall_conflict_rate=overall_conflict_rate,
            total_samples=total_samples,
            samples_with_conflicts=samples_with_conflicts,
            total_comparisons=total_comparisons,
        )

    def _calculate_accuracy_metrics(
        self, results: List[DataSample]
    ) -> Dict[str, float]:
        """Calculate accuracy metrics (based on chosen vs rejected comparison results)"""
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
                # Skip samples with evaluation errors
                if sample.metadata and sample.metadata.get("conflict_evaluation_error"):
                    continue

                # Get evaluation results
                if (
                    not sample.metadata
                    or "conflict_evaluation_result" not in sample.metadata
                ):
                    continue

                eval_result = sample.metadata["conflict_evaluation_result"]
                accuracy_data = eval_result.get("accuracy_data", {})

                # Skip samples with errors
                if "error" in accuracy_data:
                    continue

                # Accumulate chosen vs rejected comparison statistics
                total_chosen_wins += accuracy_data.get("chosen_wins", 0)
                total_chosen_losses += accuracy_data.get("chosen_losses", 0)
                total_chosen_ties += accuracy_data.get("chosen_ties", 0)
                total_chosen_vs_rejected_comparisons += accuracy_data.get(
                    "chosen_vs_rejected_comparisons", 0
                )

                # Accumulate dominance statistics
                total_dominance_samples += accuracy_data.get("chosen_dominance", 0)

                valid_count += 1

            except Exception as e:
                logger.debug(f"Error processing sample accuracy: {str(e)}")
                pass

        # Calculate overall accuracy: proportion of chosen wins in all chosen vs rejected comparisons
        accuracy = (
            total_chosen_wins / total_chosen_vs_rejected_comparisons
            if total_chosen_vs_rejected_comparisons > 0
            else 0.0
        )

        # Calculate dominance rate: proportion of samples where chosen has uniquely highest win count
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
        """Generate evaluation summary"""
        # Calculate conflict metrics
        conflict_metrics = self._calculate_conflict_metrics(results)

        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(results)

        # Calculate basic statistics
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

                # Count model calls
                extra_data = sample.metadata["conflict_evaluation_result"].get(
                    "extra_data", {}
                )
                comparison_quality = extra_data.get("comparison_quality", {})

                expected = comparison_quality.get("expected_comparisons", 0)
                successful = comparison_quality.get("successful_comparisons", 0)

                total_expected_comparisons += expected
                total_successful_comparisons += successful

                # Count model calls based on comparison mode
                if self.reward.comparison_mode == "pairwise":
                    total_model_calls += (
                        successful  # pairwise: 1 call per comparison pair
                    )
                else:  # pointwise
                    total_model_calls += (
                        successful * 2
                    )  # pointwise: 2 calls per comparison pair

        # Build summary
        return {
            "model": self.reward.llm.model if self.reward.llm else "unknown",
            "comparison_mode": self.reward.comparison_mode,
            "accuracy_metrics": accuracy_metrics,
            "conflict_metrics": {
                "overall_conflict_rate": conflict_metrics.overall_conflict_rate,
                "total_samples": conflict_metrics.total_samples,
                "samples_with_conflicts": conflict_metrics.samples_with_conflicts,
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
    """Main conflict detection evaluation pipeline, implemented in RewardBench2 style

    Supports two comparison modes: pairwise (direct pairwise comparison) and pointwise (independent scoring then comparison)

    Args:
        data_path: Input dataset file path
        result_path: Path to save evaluation results
        max_samples: Maximum number of samples to process (-1 means all)
        model: Model identifier string or configuration dictionary
        max_workers: Maximum number of parallel worker threads for evaluation
        comparison_mode: Comparison mode, "pairwise" or "pointwise"
        save_detailed_outputs: Whether to save detailed model output records
        random_seed: Random seed to ensure sampling reproducibility (only effective when max_samples != -1)
    """
    try:
        # Validate input parameters
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if comparison_mode not in ["pairwise", "pointwise"]:
            raise ValueError(
                f"Unsupported comparison mode: {comparison_mode}, supported modes: pairwise, pointwise"
            )

        # Set random seed to ensure reproducibility
        if max_samples > 0:
            # Only set random seed when sampling is needed
            random.seed(random_seed)
            np.random.seed(random_seed)
            print(f"Setting random seed: {random_seed} (for sample extraction)")

        if max_samples <= 0:
            max_samples = None  # Load all samples

        # Create data loading configuration
        config = {
            "path": data_path,
            "limit": max_samples,
        }

        # Initialize data loading module
        print(f"Loading data from: {data_path}")
        load_module = create_loader(
            name="rewardbench2",
            load_strategy_type="local",
            data_source="rewardbench2",
            config=config,
        )

        # Initialize language model for evaluation
        print(f"Initializing model: {model}")
        if isinstance(model, str):
            llm = OpenaiLLM(model=model)
        elif isinstance(model, dict):
            llm = OpenaiLLM(**model)
        else:
            raise ValueError(
                f"Invalid model type: {type(model)}. Expected str or dict type."
            )

        # Load evaluation dataset
        dataset = load_module.run()
        samples = dataset.get_data_samples()
        print(f"Loaded {len(samples)} samples for evaluation")

        if not samples:
            print("No samples loaded. Please check data file and configuration.")
            return

        # Create evaluator instance
        evaluator = ConflictDetectionEvaluator(
            reward=ConflictDetectionReward(
                name="conflict_detection",
                llm=llm,
                comparison_mode=comparison_mode,
                save_detailed_outputs=save_detailed_outputs,
            )
        )

        # Execute evaluation pipeline with parallel processing support
        results = evaluator.run(samples=samples, max_workers=max_workers)

        # Print detailed evaluation results
        print_evaluation_results(results)

        # Ensure result directory exists
        result_dir = os.path.dirname(result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        # Persist evaluation results to file
        print(f"Results saved to: {result_path}")
        write_json(results, result_path)

        print("Evaluation completed successfully!")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


def print_evaluation_results(results: dict):
    """Print detailed evaluation results, consistent with rewardbench2.py style"""
    print("\n" + "=" * 80)
    print("Conflict Detection Evaluation Results")
    print("=" * 80)

    print(f"\nModel: {results.get('model', 'Unknown')}")
    print(f"Comparison mode: {results.get('comparison_mode', 'Unknown')}")

    # Print accuracy metrics
    accuracy_metrics = results.get("accuracy_metrics", {})
    if accuracy_metrics:
        strategy = accuracy_metrics.get("strategy", "unknown")
        print("\nAccuracy Metrics (based on chosen vs rejected comparisons):")
        print(
            f"  Accuracy: {accuracy_metrics.get('accuracy', 0):.4f} ({accuracy_metrics.get('accuracy', 0)*100:.2f}%)"
        )
        print(
            f"  Chosen dominance rate: {accuracy_metrics.get('chosen_dominance_rate', 0):.4f} ({accuracy_metrics.get('chosen_dominance_rate', 0)*100:.2f}%)"
        )
        print(
            f"  Chosen wins: {accuracy_metrics.get('total_chosen_wins', 0)}/{accuracy_metrics.get('total_chosen_vs_rejected_comparisons', 0)}"
        )
        print(
            f"  Chosen losses: {accuracy_metrics.get('total_chosen_losses', 0)} ({accuracy_metrics.get('chosen_loss_rate', 0)*100:.1f}%)"
        )
        print(
            f"  Ties: {accuracy_metrics.get('total_chosen_ties', 0)} ({accuracy_metrics.get('chosen_tie_rate', 0)*100:.1f}%)"
        )
        print(
            f"  Dominant samples: {accuracy_metrics.get('total_dominance_samples', 0)}/{accuracy_metrics.get('valid_samples', 0)}"
        )
        print(
            f"  Valid samples: {accuracy_metrics.get('valid_samples', 0)}/{accuracy_metrics.get('total_samples', 0)}"
        )
        print(
            f"  Avg comparisons per sample: {accuracy_metrics.get('avg_comparisons_per_sample', 0):.1f}"
        )
        print(f"  Evaluation strategy: {strategy}")
        print(
            "  Note: Dominance rate indicates proportion of samples where chosen has uniquely highest win count"
        )

    # Print conflict metrics
    conflict_metrics = results.get("conflict_metrics", {})
    if conflict_metrics:
        print("\nConflict Metrics (based on SCC detection, lower is better):")
        print(
            f"  Overall conflict rate: {conflict_metrics.get('overall_conflict_rate', 0):.2f}%"
        )
        print("  Detection method: Tarjan's SCC algorithm")

        print("\nDetailed statistics:")
        print(f"  Total samples: {conflict_metrics.get('total_samples', 0)}")
        print(
            f"  Samples with conflicts: {conflict_metrics.get('samples_with_conflicts', 0)}"
        )
        print(f"  Total comparisons: {conflict_metrics.get('total_comparisons', 0)}")

    # Print evaluation summary
    eval_summary = results.get("evaluation_summary", {})
    if eval_summary:
        print("\nEvaluation Summary:")
        print(
            f"  Successfully evaluated samples: {eval_summary.get('successful_samples', 0)}"
        )
        print(f"  Failed samples: {eval_summary.get('failed_samples', 0)}")
        print(f"  Success rate: {eval_summary.get('success_rate', 0):.2%}")
        print(f"  Total model calls: {eval_summary.get('total_model_calls', 0)}")
        print(
            f"  Expected comparisons: {eval_summary.get('total_expected_comparisons', 0)}"
        )
        print(
            f"  Successful comparisons: {eval_summary.get('total_successful_comparisons', 0)}"
        )
        print(
            f"  Comparison success rate: {eval_summary.get('comparison_success_rate', 0):.2%}"
        )

    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"  Total samples processed: {results.get('total_count', 0)}")
    print(f"  Non-Ties samples: {results.get('non_ties_count', 0)}")
    print(f"  Ties samples (skipped): {results.get('ties_count', 0)}")
    print(f"  Worker threads used: {results.get('max_workers', 0)}")

    # Performance explanation
    comparison_mode = results.get("comparison_mode", "unknown")
    if comparison_mode == "pairwise":
        print(
            "  Comparison strategy: Direct pairwise comparison (1 model call per comparison pair)"
        )
    elif comparison_mode == "pointwise":
        print(
            "  Comparison strategy: Independent scoring then comparison (2 model calls per comparison pair)"
        )

    # Conflict metric interpretation
    if conflict_metrics.get("overall_conflict_rate", 0) > 0:
        print("\n📊 Conflict Analysis:")
        overall_rate = conflict_metrics.get("overall_conflict_rate", 0)
        if overall_rate < 5:
            assessment = "Very Low - Excellent performance"
        elif overall_rate < 15:
            assessment = "Low - Good performance"
        elif overall_rate < 30:
            assessment = "Medium - Room for improvement"
        else:
            assessment = "High - Logical consistency issues"
        print(f"  Assessment: {assessment}")
        print(
            f"  Interpretation: {overall_rate:.2f}% of samples contain cycle conflicts"
        )
        print("                  (detected using Strongly Connected Components)")
    else:
        print(
            "\n✅ No conflicts detected - Model shows perfect logical consistency (no cycles)!"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
