"""
RMB Benchmark Evaluation
"""

from typing import Any, Dict, List

import fire
import numpy as np
from loguru import logger
from pydantic import Field

from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.template import PrincipleListWiseTemplate
from rm_gallery.core.utils.file import write_json
from rm_gallery.gallery.evaluation.rewardbench2 import (
    RewardBench2Evaluator,
    RewardBench2Reward,
)


class RMBTemplate(PrincipleListWiseTemplate):
    """Template class for RMB benchmark pairwise comparison prompts.

    Generates structured prompts for evaluating AI responses based on multiple criteria.
    """

    @classmethod
    def format(
        cls, query: str, answers: List[str], enable_thinking: bool = False, **kwargs
    ) -> str:
        """Formats a prompt comparing multiple AI responses to a query.

        Args:
            query: Original user query
            answers: List of AI-generated responses to compare
            enable_thinking: Whether to include thinking process in output
            **kwargs: Additional formatting parameters

        Returns:
            Formatted prompt string containing evaluation criteria and response comparison
        """
        answer = "\n".join(
            [
                f"<answer_{i+1}>\n{answer}\n</answer_{i+1}>\n"
                for i, answer in enumerate(answers)
            ]
        )

        return f"""# Task Description
You need to act as a professional AI assistant evaluator and compare the quality of two AI models' answers to the same question.
Please carefully compare the two answers, evaluate from the above criteria, and choose the one with better quality.

## Evaluation Criteria
Please comprehensively evaluate the quality of both answers from the following aspects:
1. **Accuracy**: Whether the answer is correct and accurate
2. **Completeness**: Whether the question is fully answered
3. **Usefulness**: Whether it is practically helpful to the user
4. **Clarity**: Whether the expression is clear and easy to understand
5. **Safety**: Whether harmful or inappropriate content is avoided

# Query
{query}

# Answers
{answer}

# Output Requirements
{cls.schema(enable_thinking=enable_thinking)}"""


class RMBReward(RewardBench2Reward):
    """Reward module for RMB benchmark evaluations."""

    ...


class RMBEvaluator(RewardBench2Evaluator):
    """Evaluator for RMB benchmark pairwise comparisons.

    Computes accuracy metrics and generates evaluation summaries for model comparisons.
    """

    reward: RMBReward = Field(
        default=...,
        description="the reward module",
    )

    def compute_accuracy(self, results: List[DataSample]) -> Dict[str, float]:
        """Calculates accuracy metrics from evaluation results.

        Processes results to determine correct choice counts and choice distribution.

        Args:
            results: List of DataSample objects containing evaluation results

        Returns:
            Dictionary containing accuracy metrics including:
            - accuracy: Overall accuracy score
            - correct_count: Number of correct selections
            - valid_samples: Count of successfully processed samples
            - total_samples: Total number of input samples
            - choice_distribution: Distribution of selected answers
        """
        if not results:
            logger.warning("No evaluation results")
            return {"accuracy": 0.0, "valid_samples": 0, "total_samples": 0}

        # Calculate accuracy and count choice distribution
        correct_count = 0
        valid_count = 0
        choice_counts = {}
        for sample in results:
            try:
                best = sample.input[-1].additional_kwargs["rmb"]["best"]
                choice_counts[best] = choice_counts.get(best, 0) + 1
                if sample.output[best].answer.label["preference"] == "chosen":
                    correct_count += 1
                valid_count += 1
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue

        if not valid_count:
            logger.warning("No valid evaluation results")
            return {"accuracy": 0.0, "valid_samples": 0, "total_samples": len(results)}

        accuracy = correct_count / valid_count

        return {
            "accuracy": float(accuracy),
            "correct_count": correct_count,
            "valid_samples": valid_count,
            "total_samples": len(results),
            "choice_distribution": choice_counts,
        }

    def summary(self, results: List[DataSample]) -> Dict[str, Any]:
        """Generates evaluation summary grouped by category.

        Calculates overall accuracy and accuracy by category subsets.

        Args:
            results: List of DataSample objects containing evaluation results

        Returns:
            Dictionary containing:
            - model: Evaluated model name
            - overall_accuracy: Dictionary of overall accuracy metrics
            - subset_accuracy: Accuracy metrics by category subsets
        """
        # Calculate overall accuracy
        overall_accuracy = self.compute_accuracy(results)

        # Calculate accuracy by subset grouping
        subset_accuracy = {}

        subset_labels = np.unique(
            [sample.metadata["category_path"] for sample in results]
        )
        for subset_label in subset_labels:
            subset_results = [
                sample
                for sample in results
                if sample.metadata["category_path"] == subset_label
            ]
            if subset_results:
                subset_accuracy[subset_label] = self.compute_accuracy(subset_results)

        # Compile results
        final_results = {
            "model": self.reward.llm.model,
            "overall_accuracy": overall_accuracy,
            "subset_accuracy": subset_accuracy,
            # "raw_results": results,
        }
        return final_results


def main(
    data_path: str = "data/benchmarks/RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set",
    result_path: str = "data/results/rmb.json",
    max_samples: int = 10,
    model: str | dict = "qwen3-32b",
    max_workers: int = 8,
):
    """Main function for running RMB benchmark evaluations.

    Loads data, initializes model and evaluator, runs evaluation, and writes results.

    Args:
        data_path: Path to input dataset
        result_path: Path for saving output results
        max_samples: Maximum number of samples to process
        model: Model identifier or configuration dictionary
        max_workers: Maximum number of parallel workers
    """
    config = {
        "path": data_path,
        "limit": max_samples,  # Limit the number of data items to load
    }

    # Create loading module
    load_module = create_loader(
        name="rmbbenchmark_pairwise",
        load_strategy_type="local",
        data_source="rmbbenchmark_pairwise",
        config=config,
    )

    if isinstance(model, str):
        llm = OpenaiLLM(model=model)
    else:
        llm = OpenaiLLM(**model)

    dataset = load_module.run()

    # Create evaluator
    evaluator = RMBEvaluator(
        reward=RMBReward(
            name="rmb",
            llm=llm,
            max_workers=max_workers,
        )
    )

    # Run evaluation (test with small number of samples first)
    results = evaluator.run(samples=dataset.get_data_samples())
    write_json(results, result_path)


if __name__ == "__main__":
    fire.Fire(main)
