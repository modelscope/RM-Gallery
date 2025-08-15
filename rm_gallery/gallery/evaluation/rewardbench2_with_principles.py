"""
RewardBench2 Evaluator with Principle Support

This module provides an enhanced version of RewardBench2 evaluation that supports
custom evaluation principles for more targeted assessment.
"""
import os
import fire
from typing import Dict, List
from pydantic import Field

from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.utils.file import write_json
from rm_gallery.gallery.evaluation.rewardbench2 import RewardBench2Evaluator
from rm_gallery.gallery.evaluation.template import RewardBench2PrincipleReward


class RewardBench2PrincipleEvaluator(RewardBench2Evaluator):
    """Enhanced RewardBench2 Evaluator with principle support.
    
    Extends the base RewardBench2Evaluator to support custom evaluation principles
    while maintaining compatibility with the original evaluation protocol.
    """
    
    # Override the reward field to accept RewardBench2PrincipleReward
    reward: RewardBench2PrincipleReward = Field(
        default=...,
        description="the reward module with principle support",
    )

    def __init__(self, reward: RewardBench2PrincipleReward, **kwargs):
        """Initialize evaluator with principle-aware reward model."""
        super().__init__(reward=reward, **kwargs)


def main(
    data_path: str = "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path: str = "data/results/rewardbench2_with_principles.json",
    max_samples: int = -1,
    model: str | dict = "qwen2.5-72b-instruct",
    max_workers: int = 8,
    principles: str = "",
    principles_file: str = "",
):
    """Main evaluation pipeline with principle support.

    Args:
        data_path: Path to input dataset file
        result_path: Path for saving evaluation results
        max_samples: Maximum number of samples to process (-1 for all)
        model: Model identifier string or configuration dictionary
        max_workers: Maximum number of parallel workers for evaluation
        principles: Custom evaluation principles as string
        principles_file: Path to file containing evaluation principles
    """
    try:
        # Validate input parameters
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if max_samples <= 0:
            max_samples = None  # Load all samples

        # Load principles from file if specified
        final_principles = principles
        if principles_file and os.path.exists(principles_file):
            with open(principles_file, 'r', encoding='utf-8') as f:
                file_principles = f.read().strip()
            if file_principles:
                final_principles = file_principles
                print(f"Loaded principles from: {principles_file}")

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
            raise ValueError(f"Invalid model type: {type(model)}. Expected str or dict.")

        # Load evaluation dataset
        dataset = load_module.run()
        samples = dataset.get_data_samples()
        print(f"Loaded {len(samples)} samples for evaluation")

        if not samples:
            print("No samples loaded. Check data file and configuration.")
            return

        # Display principles being used
        if final_principles:
            print(f"\nUsing custom evaluation principles:")
            print("-" * 40)
            print(final_principles)
            print("-" * 40)
        else:
            print("\nUsing default evaluation criteria (no custom principles)")

        # Create evaluator instance with principle support
        evaluator = RewardBench2PrincipleEvaluator(
            reward=RewardBench2PrincipleReward(
                name="rewardbench2_with_principles",
                llm=llm,
                principles=final_principles
            )
        )

        # Execute evaluation pipeline with parallel processing
        results = evaluator.run(samples=samples, max_workers=max_workers)

        # Print detailed evaluation results
        print("\n" + "="*80)
        print("EVALUATION RESULTS (WITH PRINCIPLES)")
        print("="*80)

        print(f"\nModel: {results.get('model', 'Unknown')}")

        # Print overall accuracy
        overall_acc = results.get('overall_accuracy', {})
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {overall_acc.get('accuracy', 0):.4f} ({overall_acc.get('accuracy', 0)*100:.2f}%)")
        print(f"  Correct: {overall_acc.get('correct_count', 0)}/{overall_acc.get('valid_samples', 0)}")
        print(f"  Total samples: {overall_acc.get('total_samples', 0)}")
        print(f"  Non-Ties samples: {overall_acc.get('non_ties_samples', 0)}")
        print(f"  Ties samples: {overall_acc.get('ties_samples', 0)}")

        # Print subset accuracy
        subset_acc = results.get('subset_accuracy', {})
        if subset_acc:
            print(f"\nSubset Performance:")
            for subset, metrics in subset_acc.items():
                accuracy = metrics.get('accuracy', 0)
                correct = metrics.get('correct_count', 0)
                valid = metrics.get('valid_samples', 0)
                total = metrics.get('total_samples', 0)
                print(f"  {subset:15s}: {accuracy:.4f} ({accuracy*100:5.2f}%) - {correct:2d}/{valid:2d} correct, {total:2d} total")

        print("\n" + "="*80)

        # Ensure result directory exists
        result_dir = os.path.dirname(result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        
        # Add principles information to results
        results["evaluation_config"] = {
            "principles_used": final_principles,
            "principles_source": "file" if principles_file else "parameter",
            "has_custom_principles": bool(final_principles)
        }
        
        # Persist evaluation results to file
        print(f"Results saved to: {result_path}")
        write_json(results, result_path)
        
        print("Evaluation with principles completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


def create_sample_principles_file(output_path: str = "sample_principles.txt"):
    """Create a sample principles file for demonstration."""
    sample_principles = """1. Accuracy and Factual Correctness: The response should provide accurate, factual information without errors or misleading statements.

2. Completeness and Relevance: The response should fully address the user's question with relevant information that directly answers what was asked.

3. Clarity and Coherence: The response should be well-structured, easy to understand, and logically organized with clear explanations.

4. Helpfulness and Practicality: The response should provide actionable insights or useful information that helps the user achieve their goal.

5. Safety and Harmlessness: The response should avoid harmful, inappropriate, or potentially dangerous content while maintaining ethical standards."""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_principles)
    
    print(f"Sample principles file created at: {output_path}")
    print("You can edit this file and use it with --principles_file parameter")


if __name__ == "__main__":
    fire.Fire({
        "evaluate": main,
        "create_sample_principles": create_sample_principles_file
    })