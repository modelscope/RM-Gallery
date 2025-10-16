#!/usr/bin/env python3
"""
Rubric Analysis Framework - Core Module

Integrates functionality from llm_eva.py and anla.py into a cohesive system
that leverages base.py components for consistency and reusability.

Core Ideas:
1. Evaluation: Direct evaluation using RubricEvaluationTemplate (aligned with generator.py)
2. Comprehensive Metrics: Coverage, Precision, Contribution analysis
3. Optimization Strategies: Sampling and clustering for efficiency
4. Template Reuse: Leverage base.py prompt templates
"""

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.message import format_messages
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.utils.file import read_jsonl

from .base import RubricEvaluationTemplate


@dataclass
class RubricMetrics:
    """Data class for rubric evaluation metrics"""

    coverage: float
    precision: float  # Also called selectivity
    contribution: float
    covered_samples: int
    total_samples: int
    correct_predictions: int
    rubric_text: str
    rubric_index: int
    rubric_type: str = "unknown"  # "source" or "target"


@dataclass
class EvaluationConfig:
    """Configuration for rubric evaluation"""

    model: str = "qwen3-32b"
    max_workers: int = 256
    enable_thinking: bool = True
    optimization_strategy: str = "sampling"  # "sampling", "clustering", "none"
    source_sample_ratio: float = 0.2
    target_sample_ratio: float = 1.0
    contribution_sample_ratio: float = 0.5
    max_tokens: int = 2048
    thinking_budget: int = 2048


class RubricAnalyzer:
    """
    Rubric analyzer that combines evaluation and diagnostic capabilities

    Features:
    - Uses RubricEvaluationTemplate for consistent evaluation (aligned with generator.py)
    - Supports both individual and ensemble rubric analysis
    - Provides comprehensive metrics (Coverage, Precision, Contribution)
    - Implements optimization strategies for large-scale analysis
    """

    def __init__(self, config: EvaluationConfig = None):
        """Initialize the analyzer"""
        self.config = config or EvaluationConfig()

        # Initialize LLM
        self.llm = OpenaiLLM(
            model=self.config.model,
            enable_thinking=self.config.enable_thinking,
            max_tokens=self.config.max_tokens,
            thinking_budget=self.config.thinking_budget,
            stop_if_detect_repetition=True,
        )

        # Cache for evaluation results
        self._evaluation_cache = {}

        logger.info(f"Initialized Rubric Analyzer with model {self.config.model}")
        logger.info(f"Optimization strategy: {self.config.optimization_strategy}")

    def load_dataset(
        self, dataset_path: str, domains: List[str] = None, max_samples: int = -1
    ) -> List[DataSample]:
        """Load preference dataset with filtering options"""
        raw_samples = read_jsonl(dataset_path)

        # Filter by domains if specified
        if domains:
            samples = [
                DataSample(**sample)
                for sample in raw_samples
                if sample["metadata"]["domain"] in domains
            ]
            logger.info(
                f"Filtered by domains {domains}: {len(samples)}/{len(raw_samples)} samples"
            )
        else:
            samples = [DataSample(**sample) for sample in raw_samples]

        # Set preference labels (following main.py transform logic)
        for sample in samples:
            for output in sample.output:
                output.answer.label["preference"] = (
                    "chosen" if output.answer.label["is_preferred"] else "rejected"
                )

        # Filter out tie data
        samples = [
            sample
            for sample in samples
            if sample.metadata.get("overall_preference", 0) != 0
        ]

        # Limit samples if specified
        if max_samples > 0:
            samples = samples[:max_samples]

        logger.info(f"Final dataset: {len(samples)} samples from {dataset_path}")
        return samples

    def get_ground_truth_preference(self, sample: DataSample) -> str:
        """Extract ground truth preference from sample"""
        overall_pref = sample.metadata.get("overall_preference", 0)
        if overall_pref < 0:
            return "A>B"
        elif overall_pref > 0:
            return "B>A"
        else:
            # Check individual preferences
            outputs = sample.output
            if len(outputs) >= 2:
                pref_a = outputs[0].answer.label.get("is_preferred", False)
                pref_b = outputs[1].answer.label.get("is_preferred", False)

                if pref_a and not pref_b:
                    return "A>B"
                elif pref_b and not pref_a:
                    return "B>A"

            return "A=B"

    def optimize_evaluation_data(
        self, rubrics: List[str], dataset: List[DataSample], rubric_type: str = "source"
    ) -> Tuple[List[str], List[DataSample]]:
        """Apply optimization strategies to reduce computational complexity"""

        if self.config.optimization_strategy == "none":
            return rubrics, dataset

        # Determine sample ratio based on rubric type
        if rubric_type == "source":
            sample_ratio = self.config.source_sample_ratio
        else:
            sample_ratio = self.config.target_sample_ratio

        # Dataset sampling
        sample_size = max(50, int(len(dataset) * sample_ratio))
        optimized_dataset = random.sample(dataset, min(sample_size, len(dataset)))

        # Rubric optimization
        if self.config.optimization_strategy == "sampling":
            optimized_rubrics = self._sample_rubrics(rubrics, rubric_type)
        elif self.config.optimization_strategy == "clustering":
            optimized_rubrics = self._cluster_rubrics(rubrics)
        else:
            optimized_rubrics = rubrics

        logger.info(
            f"Optimization ({rubric_type}): {len(optimized_rubrics)} rubrics, {len(optimized_dataset)} samples"
        )
        return optimized_rubrics, optimized_dataset

    def _sample_rubrics(self, rubrics: List[str], rubric_type: str) -> List[str]:
        """Sample rubrics for efficiency"""
        if rubric_type == "source" and len(rubrics) > 50:
            sample_size = min(50, len(rubrics))
            return random.sample(rubrics, sample_size)
        return rubrics

    def _cluster_rubrics(self, rubrics: List[str]) -> List[str]:
        """Cluster rubrics by complexity (simplified version)"""
        # Group by length as a proxy for complexity
        short_rubrics = [r for r in rubrics if len(r) < 100]
        medium_rubrics = [r for r in rubrics if 100 <= len(r) < 200]
        long_rubrics = [r for r in rubrics if len(r) >= 200]

        # Sample from each group
        max_per_group = 20
        selected_rubrics = []
        for group in [short_rubrics, medium_rubrics, long_rubrics]:
            if group:
                selected_rubrics.extend(
                    random.sample(group, min(max_per_group, len(group)))
                )

        return selected_rubrics

    def evaluate_single_rubric(
        self,
        rubric: str,
        dataset: List[DataSample],
        rubric_index: int = 0,
        rubric_type: str = "unknown",
    ) -> RubricMetrics:
        """
        Evaluate a single rubric using RubricEvaluationTemplate with multithreading

        This method uses the same evaluation logic as generator.py
        for consistent evaluation across the framework.
        """
        logger.info(
            f"Evaluating {rubric_type} rubric {rubric_index + 1}: {rubric[:100]}..."
        )

        covered_samples = 0
        correct_predictions = 0
        total_samples = len(dataset)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(
                    self._evaluate_sample_with_ground_truth, rubric, sample
                ): sample
                for sample in dataset
            }

            # Collect results with progress tracking
            for future in tqdm(
                as_completed(future_to_sample),
                total=len(dataset),
                desc=f"Evaluating rubric {rubric_index + 1}",
            ):
                try:
                    provides_signal, is_correct = future.result()
                    if provides_signal:
                        covered_samples += 1
                        if is_correct:
                            correct_predictions += 1
                except Exception as e:
                    logger.error(f"Error evaluating sample: {e}")
                    continue

        # Calculate metrics
        coverage = covered_samples / total_samples if total_samples > 0 else 0.0
        precision = (
            correct_predictions / covered_samples if covered_samples > 0 else 0.0
        )

        return RubricMetrics(
            coverage=coverage,
            precision=precision,
            contribution=0.0,  # Will be calculated separately
            covered_samples=covered_samples,
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            rubric_text=rubric,
            rubric_index=rubric_index,
            rubric_type=rubric_type,
        )

    def _evaluate_sample_with_ground_truth(
        self, rubric: str, sample: DataSample
    ) -> Tuple[bool, bool]:
        """Helper function to evaluate a sample and compare with ground truth"""
        try:
            # Use generator.py style evaluation with RubricEvaluationTemplate
            # Format rubrics string
            rubrics_str = f"1. {rubric}"

            # Get query and answers
            query = format_messages(sample.input)
            answers = [output.answer.content for output in sample.output]

            # Only support pairwise comparison
            if len(answers) != 2:
                logger.warning(
                    "Evaluation only supports pairwise comparison (2 answers)"
                )
                return False, False

            # Use template to format prompt
            prompt = RubricEvaluationTemplate.format(
                query=query,
                response_a=answers[0],
                response_b=answers[1],
                rubrics=rubrics_str,
                enable_thinking=self.llm.enable_thinking
                if hasattr(self.llm, "enable_thinking")
                else False,
            )

            # Get LLM response
            response = self.llm.simple_chat(query=prompt)

            # Parse using template
            parsed = RubricEvaluationTemplate.parse(response)
            preference = parsed.preference.upper()

            # Extract evaluation results
            ground_truth = self.get_ground_truth_preference(sample)
            (
                provides_signal,
                prediction,
            ) = self._extract_evaluation_result_from_preference(
                preference, ground_truth
            )

            is_correct = False
            if provides_signal and prediction == ground_truth:
                is_correct = True

            return provides_signal, is_correct

        except Exception as e:
            logger.error(f"Error evaluating sample: {e}")
            return False, False

    def _extract_evaluation_result_from_preference(
        self, preference: str, ground_truth: str
    ) -> Tuple[bool, str]:
        """Extract evaluation result from preference string (generator.py style)"""
        try:
            # Convert preference to standard format (same logic as generator.py)
            if preference == "A" or "RESPONSE A" in preference:
                prediction = "A>B"
                provides_signal = True
            elif preference == "B" or "RESPONSE B" in preference:
                prediction = "B>A"
                provides_signal = True
            elif preference == "TIE" or "EQUAL" in preference:
                prediction = "A=B"
                provides_signal = False  # Tie means no discriminative signal
            else:
                prediction = "A=B"
                provides_signal = False

            return provides_signal, prediction

        except Exception as e:
            logger.error(f"Error extracting evaluation result: {e}")
            return False, "A=B"

    def _extract_evaluation_result(
        self, evaluated_sample: DataSample, ground_truth: str
    ) -> Tuple[bool, str]:
        """Extract evaluation result from evaluated sample (legacy method)"""
        try:
            # Get preference from metadata (set by base.py RubricEvaluator)
            preference = evaluated_sample.metadata.get("preference", "TIE")
            return self._extract_evaluation_result_from_preference(
                preference, ground_truth
            )

        except Exception as e:
            logger.error(f"Error extracting evaluation result: {e}")
            return False, "A=B"

    def calculate_ensemble_accuracy(
        self, rubrics: List[str], dataset: List[DataSample]
    ) -> float:
        """Calculate accuracy using ensemble of all rubrics with multithreading"""
        if not rubrics:
            return 0.0

        correct_predictions = 0
        total_samples = len(dataset)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self._evaluate_ensemble_sample, rubrics, sample): sample
                for sample in dataset
            }

            # Collect results with progress tracking
            for future in tqdm(
                as_completed(future_to_sample),
                total=len(dataset),
                desc="Ensemble evaluation",
            ):
                try:
                    is_correct = future.result()
                    if is_correct:
                        correct_predictions += 1
                except Exception as e:
                    logger.error(f"Error in ensemble evaluation: {e}")
                    continue

        return correct_predictions / total_samples if total_samples > 0 else 0.0

    def _evaluate_ensemble_sample(self, rubrics: List[str], sample: DataSample) -> bool:
        """Helper function to evaluate a sample with ensemble of all rubrics"""
        try:
            # Use generator.py style evaluation with RubricEvaluationTemplate
            # Format rubrics string
            rubrics_str = "\n".join(
                [f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics)]
            )

            # Get query and answers
            query = format_messages(sample.input)
            answers = [output.answer.content for output in sample.output]

            # Only support pairwise comparison
            if len(answers) != 2:
                logger.warning(
                    "Ensemble evaluation only supports pairwise comparison (2 answers)"
                )
                return False

            # Use template to format prompt
            prompt = RubricEvaluationTemplate.format(
                query=query,
                response_a=answers[0],
                response_b=answers[1],
                rubrics=rubrics_str,
                enable_thinking=self.llm.enable_thinking
                if hasattr(self.llm, "enable_thinking")
                else False,
            )

            # Get LLM response
            response = self.llm.simple_chat(query=prompt)

            # Parse using template
            parsed = RubricEvaluationTemplate.parse(response)
            preference = parsed.preference.upper()

            ground_truth = self.get_ground_truth_preference(sample)

            # Get ensemble prediction
            if preference == "A" or "RESPONSE A" in preference:
                prediction = "A>B"
            elif preference == "B" or "RESPONSE B" in preference:
                prediction = "B>A"
            else:
                prediction = "A=B"

            return prediction == ground_truth

        except Exception as e:
            logger.error(f"Error in ensemble sample evaluation: {e}")
            return False

    def calculate_contribution(
        self, target_rubrics: List[str], rubric_index: int, dataset: List[DataSample]
    ) -> float:
        """Calculate contribution of a specific rubric by removal with multithreading"""
        # Use simplified contribution calculation for efficiency
        sample_size = max(50, int(len(dataset) * self.config.contribution_sample_ratio))
        contribution_dataset = random.sample(dataset, min(sample_size, len(dataset)))

        # Calculate full ensemble accuracy
        full_accuracy = self.calculate_ensemble_accuracy(
            target_rubrics, contribution_dataset
        )

        # Calculate accuracy without the target rubric
        remaining_rubrics = [
            r for i, r in enumerate(target_rubrics) if i != rubric_index
        ]
        if remaining_rubrics:
            reduced_accuracy = self._calculate_reduced_accuracy(
                remaining_rubrics, contribution_dataset
            )
        else:
            reduced_accuracy = 0.5  # Random baseline

        contribution = full_accuracy - reduced_accuracy
        return contribution

    def _calculate_reduced_accuracy(
        self, remaining_rubrics: List[str], dataset: List[DataSample]
    ) -> float:
        """Calculate accuracy with reduced rubric set using multithreading"""
        correct_predictions = 0
        total_samples = len(dataset)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(
                    self._evaluate_contribution_sample, remaining_rubrics, sample
                ): sample
                for sample in dataset
            }

            # Collect results
            for future in tqdm(
                as_completed(future_to_sample),
                total=len(dataset),
                desc="Calculating contribution",
            ):
                try:
                    is_correct = future.result()
                    if is_correct:
                        correct_predictions += 1
                except Exception as e:
                    logger.error(f"Error in contribution calculation: {e}")
                    continue

        return correct_predictions / total_samples if total_samples > 0 else 0.0

    def _evaluate_contribution_sample(
        self, remaining_rubrics: List[str], sample: DataSample
    ) -> bool:
        """Helper function to evaluate a sample with remaining rubrics for contribution calculation"""
        try:
            # Use generator.py style evaluation with RubricEvaluationTemplate
            # Format rubrics string
            rubrics_str = "\n".join(
                [f"{i + 1}. {rubric}" for i, rubric in enumerate(remaining_rubrics)]
            )

            # Get query and answers
            query = format_messages(sample.input)
            answers = [output.answer.content for output in sample.output]

            # Only support pairwise comparison
            if len(answers) != 2:
                logger.warning(
                    "Contribution evaluation only supports pairwise comparison (2 answers)"
                )
                return False

            # Use template to format prompt
            prompt = RubricEvaluationTemplate.format(
                query=query,
                response_a=answers[0],
                response_b=answers[1],
                rubrics=rubrics_str,
                enable_thinking=self.llm.enable_thinking
                if hasattr(self.llm, "enable_thinking")
                else False,
            )

            # Get LLM response
            response = self.llm.simple_chat(query=prompt)

            # Parse using template
            parsed = RubricEvaluationTemplate.parse(response)
            preference = parsed.preference.upper()

            ground_truth = self.get_ground_truth_preference(sample)

            # Get prediction
            if preference == "A" or "RESPONSE A" in preference:
                prediction = "A>B"
            elif preference == "B" or "RESPONSE B" in preference:
                prediction = "B>A"
            else:
                prediction = "A=B"

            return prediction == ground_truth

        except Exception as e:
            logger.error(f"Error in contribution sample evaluation: {e}")
            return False

    def evaluate_rubric_set(
        self,
        rubrics: List[str],
        dataset: List[DataSample],
        rubric_type: str = "target",
        calculate_contribution: bool = True,
        parallel_rubrics: bool = True,
    ) -> Tuple[float, List[RubricMetrics]]:
        """
        Evaluate a complete set of rubrics

        Args:
            rubrics: List of rubrics to evaluate
            dataset: Dataset for evaluation
            rubric_type: Type of rubrics ("source" or "target")
            calculate_contribution: Whether to calculate contribution metrics
            parallel_rubrics: Whether to evaluate rubrics in parallel (recommended for large sets)

        Returns:
            (ensemble_accuracy, individual_metrics)
        """
        logger.info(f"Evaluating {len(rubrics)} {rubric_type} rubrics...")

        # Apply optimization
        optimized_rubrics, optimized_dataset = self.optimize_evaluation_data(
            rubrics, dataset, rubric_type
        )

        # Evaluate individual rubrics
        if parallel_rubrics and len(optimized_rubrics) > 1:
            # Parallel evaluation for multiple rubrics
            logger.info(
                f"Using parallel evaluation for {len(optimized_rubrics)} rubrics..."
            )
            individual_metrics = self._evaluate_rubrics_parallel(
                optimized_rubrics, optimized_dataset, rubric_type
            )
        else:
            # Sequential evaluation (original behavior)
            individual_metrics = []
            for i, rubric in enumerate(optimized_rubrics):
                metrics = self.evaluate_single_rubric(
                    rubric, optimized_dataset, i, rubric_type
                )
                individual_metrics.append(metrics)

        # Calculate ensemble accuracy
        ensemble_accuracy = self.calculate_ensemble_accuracy(
            optimized_rubrics, optimized_dataset
        )

        # Calculate contributions for target rubrics
        if (
            calculate_contribution
            and rubric_type == "target"
            and len(optimized_rubrics) <= 10
        ):
            logger.info("Calculating contribution metrics...")
            for i, metrics in enumerate(individual_metrics):
                contribution = self.calculate_contribution(
                    optimized_rubrics, i, optimized_dataset
                )
                metrics.contribution = contribution

        return ensemble_accuracy, individual_metrics

    def _evaluate_rubrics_parallel(
        self,
        rubrics: List[str],
        dataset: List[DataSample],
        rubric_type: str = "unknown",
    ) -> List[RubricMetrics]:
        """
        Evaluate multiple rubrics in parallel

        This is especially useful for evaluating large numbers of source rubrics
        where we don't need contribution calculations.
        """
        metrics_list = [None] * len(rubrics)

        # Use ThreadPoolExecutor to evaluate rubrics in parallel
        with ThreadPoolExecutor(
            max_workers=min(self.config.max_workers, len(rubrics))
        ) as executor:
            # Submit all rubric evaluation tasks
            future_to_index = {
                executor.submit(
                    self.evaluate_single_rubric, rubric, dataset, i, rubric_type
                ): i
                for i, rubric in enumerate(rubrics)
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_index),
                total=len(rubrics),
                desc=f"Evaluating {rubric_type} rubrics in parallel",
            ):
                try:
                    idx = future_to_index[future]
                    metrics = future.result()
                    metrics_list[idx] = metrics
                except Exception as e:
                    logger.error(f"Error evaluating rubric {idx}: {e}")
                    # Create empty metrics on error
                    idx = future_to_index[future]
                    metrics_list[idx] = RubricMetrics(
                        coverage=0.0,
                        precision=0.0,
                        contribution=0.0,
                        covered_samples=0,
                        total_samples=len(dataset),
                        correct_predictions=0,
                        rubric_text=rubrics[idx] if idx < len(rubrics) else "",
                        rubric_index=idx,
                        rubric_type=rubric_type,
                    )

        return metrics_list

    def save_analysis_results(
        self,
        ensemble_accuracy: float,
        source_metrics: List[RubricMetrics],
        target_metrics: List[RubricMetrics],
        output_path: str = "analysis_results.json",
    ):
        """Save comprehensive analysis results"""

        results = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "model": self.config.model,
                "optimization_strategy": self.config.optimization_strategy,
                "max_workers": self.config.max_workers,
                "sample_ratios": {
                    "source": self.config.source_sample_ratio,
                    "target": self.config.target_sample_ratio,
                    "contribution": self.config.contribution_sample_ratio,
                },
            },
            "ensemble_accuracy": ensemble_accuracy,
            "source_rubrics": {
                "count": len(source_metrics),
                "avg_coverage": np.mean([m.coverage for m in source_metrics])
                if source_metrics
                else 0,
                "avg_precision": np.mean([m.precision for m in source_metrics])
                if source_metrics
                else 0,
                "details": [
                    {
                        "index": m.rubric_index,
                        "coverage": m.coverage,
                        "precision": m.precision,
                        "contribution": m.contribution,
                        "rubric_preview": m.rubric_text[:100] + "..."
                        if len(m.rubric_text) > 100
                        else m.rubric_text,
                    }
                    for m in source_metrics
                ],
            },
            "target_rubrics": {
                "count": len(target_metrics),
                "avg_coverage": np.mean([m.coverage for m in target_metrics])
                if target_metrics
                else 0,
                "avg_precision": np.mean([m.precision for m in target_metrics])
                if target_metrics
                else 0,
                "avg_contribution": np.mean(
                    [m.contribution for m in target_metrics if m.contribution != 0.0]
                )
                if target_metrics
                else 0,
                "details": [
                    {
                        "index": m.rubric_index,
                        "coverage": m.coverage,
                        "precision": m.precision,
                        "contribution": m.contribution,
                        "rubric_text": m.rubric_text,
                    }
                    for m in target_metrics
                ],
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis results saved to: {output_path}")
