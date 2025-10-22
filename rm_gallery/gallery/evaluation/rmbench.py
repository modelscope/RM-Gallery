"""
RM-Bench Evaluation
"""

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Type

import fire
import numpy as np
from loguru import logger
from pydantic import Field

from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.base import BaseLLMReward, BasePairWiseReward
from rm_gallery.core.reward.schema import RewardDimensionWithRank, RewardResult
from rm_gallery.core.reward.template import RubricListWiseTemplate
from rm_gallery.core.utils.file import write_json
from rm_gallery.gallery.evaluation.rewardbench2 import RewardBench2Evaluator


class RMBenchTemplate(RubricListWiseTemplate):
    """
    Template class for generating RM-Bench evaluation prompts.

    Generates structured prompts for pairwise comparison of responses against
    multiple quality dimensions including accuracy, usefulness, safety, and style.
    """

    @classmethod
    def format(
        cls, query: str, answers: List[str], enable_thinking: bool = False, **kwargs
    ) -> str:
        """
        Generate a comparison prompt for multiple responses to a query.

        Args:
            query: Input query text
            answers: List of response texts to compare
            enable_thinking: Whether to include reasoning in output format
            **kwargs: Additional template parameters

        Returns:
            Formatted prompt string for LLM evaluation
        """
        answer = "\n".join(
            [
                f"<answer_{i+1}>\n{answer}\n</answer_{i+1}>\n"
                for i, answer in enumerate(answers)
            ]
        )

        return f"""Please act as a reward model and select the better quality response from the following two responses.
Please compare these two responses based on accuracy, usefulness, safety and style appropriateness and select the better quality one.

# Query
{query}

# Answers
{answer}
# Output Requirements
{cls.schema(enable_thinking=enable_thinking)}
"""


class RMBenchReward(BaseLLMReward, BasePairWiseReward):
    """
    Reward model implementation for RM-Bench evaluations.

    Combines LLM-based pairwise comparisons with parallel execution capabilities.
    """

    template: Type[RMBenchTemplate] = Field(
        default=RMBenchTemplate,
        description="the template to generate the prompt",
    )

    def _before_evaluate(self, sample: DataSample, **kwargs) -> Dict:
        """
        Prepares list-wise evaluation parameters.

        Extracts query and responses from the sample for comparison.

        Args:
            sample: Multi-response sample to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing query and answers for template rendering
        """
        params = super()._before_evaluate(sample=sample, **kwargs)
        answers = [output.answer.content for output in sample.output]
        params["query"] = sample.input[-1].content
        params["answers"] = answers
        return params

    def _after_evaluate(
        self, response: RubricListWiseTemplate, sample: DataSample, **kwargs
    ) -> RewardResult:
        """
        Converts LLM response to list-wise ranking metrics.

        Args:
            response: Parsed LLM comparison result
            sample: Original data sample
            **kwargs: Additional parameters

        Returns:
            RewardResult containing ranking information
        """
        scores = [0 for i in range(len(sample.output))]
        scores[response.best - 1] = 1
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
        """
        Execute evaluations in parallel across response pairs.

        Creates subsamples for each response pair and executes evaluations concurrently.
        Aggregates comparison results back into original response objects.

        Args:
            func: Evaluation function to execute
            sample: Original data sample containing multiple responses
            thread_pool: Optional thread pool executor
            **kwargs: Additional parameters

        Returns:
            Modified data sample with aggregated evaluation results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Create a deep copy to avoid modifying original sample
        sample = sample.model_copy(deep=True)

        chosen_outputs = [
            o for o in sample.output if o.answer.label["preference"] == "chosen"
        ]
        rejected_outputs = [
            o for o in sample.output if o.answer.label["preference"] == "rejected"
        ]

        # Prepare all comparison tasks
        comparison_tasks = []
        for i, output_i in enumerate(chosen_outputs):
            for j, output_j in enumerate(rejected_outputs):
                # Create subsample containing only the current response pair
                output = [
                    output_i.model_copy(deep=True),
                    output_j.model_copy(deep=True),
                ]
                # Record original order before shuffle
                is_chosen_first = True
                if random.random() < 0.5:
                    output = [output[1], output[0]]
                    is_chosen_first = False

                subsample = DataSample(
                    unique_id=sample.unique_id,
                    input=sample.input,
                    output=output,
                )
                comparison_tasks.append((i, j, subsample, is_chosen_first))

        # Execute all comparisons in parallel
        use_internal_pool = thread_pool is None
        if use_internal_pool:
            # Use up to 9 workers for the 9 comparisons within each sample
            thread_pool = ThreadPoolExecutor(max_workers=min(9, self.max_workers))

        try:
            # Submit all tasks
            future_to_task = {
                thread_pool.submit(func, sample=task[2], **kwargs): task
                for task in comparison_tasks
            }

            # Collect results
            results_map = {}
            for future in as_completed(future_to_task):
                i, j, subsample, is_chosen_first = future_to_task[future]
                try:
                    result = future.result()
                    results_map[(i, j)] = (result, is_chosen_first)
                except Exception as e:
                    logger.error(f"Comparison ({i},{j}) failed: {str(e)}")
                    results_map[(i, j)] = (None, is_chosen_first)

            # Aggregate results back to original outputs
            comparison_matrix = []
            for i, output_i in enumerate(chosen_outputs):
                row_scores = []
                for j, output_j in enumerate(rejected_outputs):
                    if (i, j) in results_map:
                        result, is_chosen_first = results_map[(i, j)]
                        if result and len(result.details) > 0:
                            # Get the score for the chosen output
                            reward = result.details[0]
                            if is_chosen_first:
                                # Chosen is at index 0
                                score = (
                                    reward.rank[0]
                                    if hasattr(reward, "rank")
                                    else reward[0].score
                                )
                            else:
                                # Chosen is at index 1
                                score = (
                                    reward.rank[1]
                                    if hasattr(reward, "rank")
                                    else reward[1].score
                                )

                            # Append to output_i's reward details
                            if is_chosen_first:
                                output_i.answer.reward.details.append(
                                    reward[0]
                                    if hasattr(reward, "__getitem__")
                                    else reward
                                )
                            else:
                                output_i.answer.reward.details.append(
                                    reward[1]
                                    if hasattr(reward, "__getitem__")
                                    else reward
                                )

                            row_scores.append(score)
                        else:
                            row_scores.append(None)
                    else:
                        row_scores.append(None)

                comparison_matrix.append(row_scores)

            sample.input[-1].additional_kwargs["rmbench"] = {
                "comparison_matrix": comparison_matrix,
            }

        finally:
            if use_internal_pool:
                thread_pool.shutdown(wait=True)

        return sample


class RMBenchEvaluator(RewardBench2Evaluator):
    """
    Evaluator class for RM-Bench benchmark assessments.
    """

    reward: RMBenchReward = Field(
        default=...,
        description="the reward module",
    )

    def compute_accuracy(self, results: List[DataSample]) -> Dict[str, float]:
        """
        Calculate accuracy metrics from evaluation results.

        Computes three accuracy types based on response style complexity:
        - Hard accuracy: Simple vs complex style comparisons
        - Normal accuracy: Same style comparisons
        - Easy accuracy: Complex vs simple style comparisons

        Args:
            results: List of evaluated data samples

        Returns:
            Dictionary containing accuracy metrics and statistics
        """
        MATRIX_SIZE = 3  # 3x3 matrix
        acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))

        valid_results = 0
        for sample in results:
            if sample.input[-1].additional_kwargs.get("rmbench", None) is None:
                continue

            comparison_matrix = sample.input[-1].additional_kwargs["rmbench"][
                "comparison_matrix"
            ]

            # Check if matrix is valid
            if len(comparison_matrix) != 3:
                continue
            if any(len(row) != 3 for row in comparison_matrix):
                continue

            # Check if there are too many None values
            flat_matrix = [item for row in comparison_matrix for item in row]
            if sum(1 for item in flat_matrix if item is None) > 3:  # Allow few failures
                continue

            valid_results += 1

            # Accumulate comparison matrix (1 means chosen wins, 0 means rejected wins)
            for i in range(MATRIX_SIZE):
                for j in range(MATRIX_SIZE):
                    value = comparison_matrix[i][j]
                    if value is not None:
                        acc_matrix[i][j] += value

        if valid_results == 0:
            logger.warning("No valid evaluation results")
            return {"hard_acc": 0.0, "normal_acc": 0.0, "easy_acc": 0.0}

        # Calculate accuracy
        acc_matrix /= valid_results

        # Calculate different accuracy types
        upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count

        normal_acc = np.mean(np.diag(acc_matrix))

        lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count

        return {
            "hard_acc": float(hard_acc),
            "normal_acc": float(normal_acc),
            "easy_acc": float(easy_acc),
            "overall_acc": float(np.mean(acc_matrix)),
            "valid_samples": valid_results,
            "total_samples": len(results),
            "acc_matrix": acc_matrix.tolist(),
        }


def main(
    data_path: str = "data/benchmarks/RM-Bench/total_dataset.json",
    result_path: str = "data/results/rmbench.json",
    max_samples: int = 10,
    model: str | dict = "qwen3-32b",
    max_workers: int = 8,
):
    """
    Main execution function for RM-Bench evaluation.

    Loads data, initializes model, runs evaluation, and saves results.

    Args:
        data_path: Path to input dataset
        result_path: Path to save evaluation results
        max_samples: Maximum number of samples to evaluate
        model: Model identifier or configuration dictionary
        max_workers: Number of parallel workers for evaluation
    """
    config = {
        "path": data_path,
        "limit": max_samples,
    }

    load_module = create_loader(
        name="rmbench",
        load_strategy_type="local",
        data_source="rmbench",
        config=config,
    )

    if isinstance(model, str):
        llm = OpenaiLLM(model=model)
    else:
        llm = OpenaiLLM(**model)

    dataset = load_module.run()

    evaluator = RMBenchEvaluator(
        reward=RMBenchReward(
            name="rmbench",
            llm=llm,
            max_workers=max_workers,
        )
    )

    results = evaluator.run(samples=dataset.get_data_samples())
    write_json(results, result_path)


if __name__ == "__main__":
    fire.Fire(main)
