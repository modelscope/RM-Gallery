"""
Conflict Detector
"""

import random
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from enum import Enum
from typing import Any, Callable, Dict, List, Type

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
from rm_gallery.core.reward.template import PrincipleListWiseTemplate
from rm_gallery.core.utils.file import write_json


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


class PairComparisonTemplate(PrincipleListWiseTemplate):
    """Template for pairwise comparison of AI responses.

    Attributes:
        best: Index of better response (0 for tie, 1/2 for winner)
    """

    best: int = Field(
        default=...,
        description="Which answer is the best? If both answers are of equal quality, please answer 0. Otherwise, give the number of best answer here!!!",
    )

    @classmethod
    def format(
        cls, query: str, answers: List[str], enable_thinking: bool = False, **kwargs
    ) -> str:
        """Formats comparison prompt for LLM evaluation.

        Args:
            query: Original question text
            answers: List of two responses to compare
            enable_thinking: Whether to include reasoning in output
            **kwargs: Additional formatting parameters

        Returns:
            Formatted prompt string for pairwise comparison
        """
        answer = "\n".join(
            [
                f"<answer_{i+1}>\n{answer}\n</answer_{i+1}>\n"
                for i, answer in enumerate(answers)
            ]
        )

        return f"""Please compare the response quality of the following two AI assistants to the same question.
Please compare the quality of these two responses based on accuracy, usefulness, relevance, completeness and clarity.

# Query
{query}

# Answers
{answer}
# Output Requirements
{cls.schema(enable_thinking=enable_thinking)}"""


class PairComparisonReward(BaseLLMReward, BasePairWiseReward):
    """Pairwise comparison reward calculator using LLM evaluation.

    Attributes:
        template: Template class for prompt generation
    """

    template: Type[PairComparisonTemplate] = Field(
        default=PairComparisonTemplate,
        description="the template to generate the prompt",
    )

    def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
        """Prepares parameters for list-wise evaluation.

        Args:
            sample: Multi-response sample to evaluate
            **kwargs: Additional parameters

        Returns:
            Dictionary containing query and all responses for comparison
        """
        params = super()._before_evaluate(sample=sample, **kwargs)
        answers = [output.answer.content for output in sample.output]
        params["query"] = sample.input[-1].content
        params["answers"] = answers
        return params

    def _after_evaluate(
        self, response: PrincipleListWiseTemplate, sample: DataSample, **kwargs
    ) -> RewardResult:
        """Converts LLM response to ranking metrics.

        Args:
            response: Parsed LLM comparison result
            sample: Original evaluation sample
            **kwargs: Additional parameters

        Returns:
            RewardResult object with comparison scores
        """
        if response.best == 0:
            scores = [0, 0]
        elif response.best == 1:
            scores = [1, -1]
        elif response.best == 2:
            scores = [-1, 1]

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name, reason=response.reason, rank=scores
                )
            ],
            extra_data={"best": response.best - 1},
        )

    def _parallel(
        self,
        func: Callable,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """Executes pairwise comparisons in parallel with bidirectional comparisons.

        Args:
            func: Evaluation function to execute
            sample: Sample containing multiple responses
            thread_pool: Optional thread pool for execution
            **kwargs: Additional parameters

        Returns:
            Modified sample with comparison results
        """
        try:
            # Create a deep copy to avoid modifying original sample
            sample = sample.model_copy(deep=True)
            comparison_results = {}
            
            results = []
            
            # Iterate through all unique response pairs for bidirectional comparison
            for i, output_i in enumerate(sample.output):
                for j, output_j in enumerate(sample.output):
                    # Skip self-comparison
                    if i == j:
                        continue

                    # Create subsample containing only the current response pair
                    # Note: order matters for detecting symmetry conflicts
                    subsample = DataSample(
                        unique_id=sample.unique_id,
                        input=sample.input,
                        output=[output_i, output_j],
                    )

                    result = func(sample=subsample, thread_pool=None, **kwargs)
                    results.append((i, j, result))

            # Process results and store comparison scores
            for i, j, result in results:
                output_i = sample.output[i]
                output_j = sample.output[j]
                
                # Add reward details to responses
                if result.details:
                    reward = result.details[0]
                    output_i.answer.reward.details.append(reward)
                    
                    # Store comparison result: positive means i > j, negative means j > i
                    if hasattr(result, 'extra_data') and 'best' in result.extra_data:
                        best_idx = result.extra_data['best']
                        if best_idx == 0:  # First response (i) is better
                            comparison_results[(i, j)] = 1
                        elif best_idx == 1:  # Second response (j) is better  
                            comparison_results[(i, j)] = -1
                        else:  # Tie
                            comparison_results[(i, j)] = 0
                    else:
                        # Fallback: use reward score if available
                        comparison_results[(i, j)] = reward.score if hasattr(reward, 'score') else 0

            sample.input[-1].additional_kwargs["conflict_detector"] = {
                "comparison_results": comparison_results,
            }
        except Exception as e:
            logger.error(f"Error in PairComparisonReward: {e}")

        return sample


class ConflictDetector(BaseEvaluator):
    """Conflict Detector for identifying inconsistent comparisons.

    Attributes:
        reward: PairComparisonReward module for evaluation
    """

    reward: PairComparisonReward = Field(
        default=...,
        description="the reward module",
    )

    def detect_symmetry_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """Detect symmetry conflicts in comparison matrix.

        Args:
            matrix: NxN comparison matrix where matrix[i][j] = score of i vs j

        Returns:
            List of symmetry conflict records
        """
        conflicts = []
        n = matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                # Check for symmetry conflicts in bidirectional comparisons
                score_ij = matrix[i][j]  # Result when comparing i vs j
                score_ji = matrix[j][i]  # Result when comparing j vs i
                
                # Skip if either comparison wasn't made
                if score_ij == 0 and score_ji == 0:
                    continue
                    
                # Symmetry conflict occurs when:
                # 1. Both comparisons favor the first option (both positive)
                # 2. Both comparisons favor the second option (both negative)
                # 3. Strong disagreement in opposite directions
                if (score_ij > 0 and score_ji > 0) or (score_ij < 0 and score_ji < 0):
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.SYMMETRY,
                            involved_items=[i, j],
                            description=f"Symmetry conflict: inconsistent comparison results between response{i} and response{j} (scores: {score_ij} vs {score_ji})",
                        )
                    )

        return conflicts

    def detect_transitivity_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """Detect transitivity conflicts in comparison matrix.

        Args:
            matrix: NxN comparison matrix

        Returns:
            List of transitivity conflict records
        """
        conflicts = []
        n = matrix.shape[0]

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        # Check transitivity: if i>j and j>k, then i should > k
                        if matrix[i][j] > 0 and matrix[j][k] > 0 and matrix[i][k] <= 0:
                            conflicts.append(
                                Conflict(
                                    conflict_type=ConflictType.TRANSITIVITY,
                                    involved_items=[i, j, k],
                                    description=f"Transitivity conflict: response{i}>response{j}>response{k}, but response{i} is not better than response{k}",
                                )
                            )

        return conflicts

    def detect_cycles(self, matrix: np.ndarray) -> List[Conflict]:
        """Detect cycle conflicts using DFS traversal.

        Args:
            matrix: NxN comparison matrix

        Returns:
            List of cycle conflict records
        """
        conflicts = []
        n = matrix.shape[0]

        def dfs_cycle_detection(node: int, path: List[int], visited: set) -> bool:
            """DFS cycle detection helper function.

            Args:
                node: Current node index
                path: Current traversal path
                visited: Set of visited nodes

            Returns:
                True if cycle found, False otherwise
            """
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycle_nodes = path[cycle_start:] + [node]
                if len(cycle_nodes) > 2:  # At least 3-node cycle
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.CYCLE,
                            involved_items=cycle_nodes[
                                :-1
                            ],  # Remove duplicate last node
                            description=f"Cycle conflict: responses {' > '.join(map(str, cycle_nodes))} form a cycle",
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

        # Detect cycles starting from each node
        for start_node in range(n):
            visited = set()
            dfs_cycle_detection(start_node, [], visited)

        # Deduplicate (same cycle may be detected multiple times)
        unique_conflicts = []
        seen_cycles = set()
        for conflict in conflicts:
            cycle_signature = tuple(sorted(conflict.involved_items))
            if cycle_signature not in seen_cycles:
                seen_cycles.add(cycle_signature)
                unique_conflicts.append(conflict)

        return unique_conflicts

    def detect_all_conflicts(self, matrix: np.ndarray) -> List[Conflict]:
        """Detect all conflict types in comparison matrix.

        Args:
            matrix: NxN comparison matrix

        Returns:
            Combined list of all detected conflicts
        """
        all_conflicts = []

        # Detect various conflicts
        all_conflicts.extend(self.detect_symmetry_conflicts(matrix))
        all_conflicts.extend(self.detect_transitivity_conflicts(matrix))
        all_conflicts.extend(self.detect_cycles(matrix))

        return all_conflicts

    def analyze_conflicts(self, sample: DataSample):
        """Analyze conflicts in comparison results.

        Args:
            sample: DataSample containing comparison results
        """
        comparison_results = sample.input[-1].additional_kwargs["conflict_detector"][
            "comparison_results"
        ]

        # Build comparison matrix with bidirectional results
        n = len(sample.output)
        comparison_matrix = np.zeros((n, n), dtype=int)

        # Fill matrix with actual bidirectional comparison results
        for (i, j), score in comparison_results.items():
            comparison_matrix[i][j] = score
            # No automatic symmetric filling since we have real bidirectional comparisons

        # Detect conflicts
        conflicts = self.detect_all_conflicts(comparison_matrix)
        sample.input[-1].additional_kwargs["conflict_detector"]["conflicts"] = conflicts
        sample.input[-1].additional_kwargs["conflict_detector"]["conflict_types"] = {
            ct.value: sum(1 for c in conflicts if c.conflict_type == ct)
            for ct in ConflictType
        }

    def summary(self, results: List[DataSample]) -> Dict[str, Any]:
        """Generate summary statistics of conflicts across samples.

        Args:
            results: List of evaluated DataSamples

        Returns:
            Dictionary containing conflict statistics
        """
        if not results:
            return {}

        valid_results = []
        for sample in results:
            self.analyze_conflicts(sample=sample)
            try:
                valid_results.append(sample)
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")

        total_samples = len(results)
        total_conflicts = sum(
            len(sample.input[-1].additional_kwargs["conflict_detector"]["conflicts"])
            for sample in valid_results
        )

        # Count conflicts by type
        conflict_counts = {ct.value: 0 for ct in ConflictType}
        for sample in valid_results:
            for conflict_type, count in (
                sample.input[-1]
                .additional_kwargs["conflict_detector"]["conflict_types"]
                .items()
            ):
                conflict_counts[conflict_type] += count

        # Calculate conflict rates
        overall_conflict_rate = (
            total_conflicts / len(valid_results) if total_samples > 0 else 0
        )
        symmetry_conflict_rate = (
            conflict_counts[ConflictType.SYMMETRY.value] / total_samples
        )
        transitivity_conflict_rate = (
            conflict_counts[ConflictType.TRANSITIVITY.value] / total_samples
        )
        cycle_conflict_rate = conflict_counts[ConflictType.CYCLE.value] / total_samples

        # Fully consistent sample ratio
        consistent_samples = sum(
            1
            for sample in valid_results
            if len(sample.input[-1].additional_kwargs["conflict_detector"]["conflicts"])
            == 0
        )
        consistent_samples_ratio = consistent_samples / total_samples

        return dict(
            symmetry_conflict_percentage=round(symmetry_conflict_rate * 100, 2),
            transitivity_conflict_percentage=round(transitivity_conflict_rate * 100, 2),
            cycle_conflict_percentage=round(cycle_conflict_rate * 100, 2),
            overall_conflict_percentage=round(overall_conflict_rate * 100, 2),
        )

    def run(self, samples: List[DataSample], **kwargs) -> dict:
        """Run conflict detection on samples.

        Args:
            samples: List of DataSamples to evaluate
            **kwargs: Additional parameters

        Returns:
            Dictionary containing evaluation results
        """
        # Avoid position bias
        for sample in samples:
            random.shuffle(sample.output)
        summary = super().run(samples, **kwargs)
        return summary


def main(
    data_path: str = "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path: str = "data/results/conflict.json",
    max_samples: int = 10,
    model: str | dict = "qwen2.5-14b-instruct",
    max_workers: int = 8,
):
    """Main function for running conflict detection pipeline.

    Args:
        data_path: Path to input data file
        result_path: Path to save output results
        max_samples: Maximum number of samples to process
        model: LLM model identifier or configuration
        max_workers: Number of parallel workers
    """
    config = {
        "path": data_path,
        "limit": max_samples,  # Limit the number of data items to load
    }

    # Create loading module
    load_module = create_loader(
        name="rewardbench2",
        load_strategy_type="local",
        data_source="rewardbench2",
        config=config,
    )

    if isinstance(model, str):
        llm = OpenaiLLM(model=model)
    else:
        llm = OpenaiLLM(**model)

    dataset = load_module.run()

    # Create evaluator
    evaluator = ConflictDetector(
        reward=PairComparisonReward(
            name="conflict-detector",
            llm=llm,
            max_workers=max_workers,
        )
    )

    # Run evaluation (test with small number of samples first)
    results = evaluator.run(samples=dataset.get_data_samples())
    write_json(results, result_path)


if __name__ == "__main__":
    fire.Fire(main)
