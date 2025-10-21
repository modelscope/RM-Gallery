#!/usr/bin/env python3
"""
Rubric Analysis Runner Script

Evaluate rubric performance on validation datasets using comprehensive metrics.
This script analyzes generated or structured rubrics to assess their quality,
coverage, precision, and contribution to ensemble performance.

This is useful for:
1. Evaluating rubric quality and effectiveness
2. Comparing different rubric sets or generation methods
3. Analyzing individual rubric contributions to ensemble performance

Features:
- Comprehensive rubric evaluation (Coverage, Precision, Contribution)
- Ensemble accuracy calculation with multiple rubrics
- Source vs. Target rubric comparison analysis
- Multithreaded evaluation for high performance
- Detailed statistics and performance metrics

"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from rm_gallery.core.reward.rubric.analyzer import EvaluationConfig, RubricAnalyzer


def load_rubrics(rubrics_path: str) -> List[str]:
    """Load rubrics from JSON file"""
    with open(rubrics_path, "r", encoding="utf-8") as f:
        rubrics = json.load(f)

    if isinstance(rubrics, list):
        return rubrics
    else:
        raise ValueError(f"Invalid rubrics format in {rubrics_path}")


def run_analysis(
    rubrics_path: str,
    dataset_path: str,
    model: str = "qwen3-32b",
    max_samples: int = 100,
    max_workers: int = 256,
    output_dir: str = None,
    source_rubrics_path: str = None,
):
    """
    Run comprehensive rubric analysis

    Args:
        rubrics_path: Path to target rubrics (main evaluation set)
        dataset_path: Path to validation dataset
        model: LLM model name for evaluation
        max_samples: Maximum samples to evaluate
        max_workers: Number of worker threads
        output_dir: Output directory for results
        source_rubrics_path: Optional path to source rubrics for comparison

    Note:
        - Target rubrics: Calculate Coverage, Precision, and Contribution
        - Source rubrics: Calculate only Coverage and Precision (for comparison baseline)
    """
    print("🔍 Running Rubric Analysis")
    print("=" * 50)

    # Load target rubrics
    rubrics = load_rubrics(rubrics_path)

    print(f"✅ Loaded {len(rubrics)} target rubrics")

    # Load source rubrics (optional)
    source_rubrics = []
    if source_rubrics_path:
        source_rubrics = load_rubrics(source_rubrics_path)
        print(f"✅ Loaded {len(source_rubrics)} source rubrics")

    print(f"🔧 Using {max_workers} worker threads for parallel processing")

    # Initialize analyzer with multithreading support
    config = EvaluationConfig(
        model=model,
        max_workers=max_workers,  # Configurable worker threads
        optimization_strategy="sampling",
        target_sample_ratio=1.0,
    )

    analyzer = RubricAnalyzer(config)

    # Load dataset
    dataset = analyzer.load_dataset(
        dataset_path, domains=["general"], max_samples=max_samples
    )

    print(f"✅ Loaded {len(dataset)} validation samples")

    # Evaluate target rubrics
    print("\n🎯 Evaluating target rubrics...")
    ensemble_accuracy, metrics = analyzer.evaluate_rubric_set(
        rubrics, dataset, "target", calculate_contribution=True
    )

    # Evaluate source rubrics (if provided)
    source_metrics = []
    if source_rubrics:
        print("\n📊 Evaluating source rubrics...")
        print(
            "   ℹ️  Note: Source rubrics only calculate Coverage and Precision (no Contribution)"
        )
        print(
            f"   🚀 Using parallel evaluation for {len(source_rubrics)} source rubrics..."
        )
        _, source_metrics = analyzer.evaluate_rubric_set(
            source_rubrics,
            dataset,
            "source",
            calculate_contribution=False,
            parallel_rubrics=True,  # Enable parallel evaluation for source rubrics
        )

    # Generate output directory name if not provided
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"rubric_analysis_results_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save results using analyzer's built-in method
    results_file = output_path / "analysis_results.json"
    analyzer.save_analysis_results(
        ensemble_accuracy, source_metrics, metrics, str(results_file)
    )

    print(f"\n💾 Results saved to: {output_path}")
    print(f"   📄 Analysis results: {results_file}")

    return ensemble_accuracy, metrics


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Simple Rubric Analysis Runner")

    # Input options
    parser.add_argument(
        "--rubrics", required=True, help="Rubrics JSON file or output directory"
    )
    parser.add_argument(
        "--dataset",
        default="./data/helpsteer3_preference_valid.jsonl",
        help="Validation dataset path",
    )
    parser.add_argument("--model", default="qwen3-32b", help="Model name")
    parser.add_argument(
        "--max-samples", type=int, default=100, help="Maximum samples for evaluation"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=256,
        help="Maximum number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for results (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--source-rubrics",
        default=None,
        help="Optional source rubrics JSON file or directory for comparison",
    )

    args = parser.parse_args()

    try:
        run_analysis(
            args.rubrics,
            args.dataset,
            args.model,
            args.max_samples,
            args.max_workers,
            args.output,
            args.source_rubrics,
        )
        print("\n🎉 Analysis completed successfully!")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
