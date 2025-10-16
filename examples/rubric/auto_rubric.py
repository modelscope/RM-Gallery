#!/usr/bin/env python3
"""
Batch Iterative Rubric Generation and MCR Information Gain Detection Pipeline

Core Ideas:
1. Batch Processing: Process only batch_size samples per iteration
2. Iterative Generation: Generate rubrics for current batch samples
3. MCR² Evaluation: Detect information gain, automatically select optimal subset
4. Adaptive Stopping: Decide whether to continue iteration based on gain threshold
5. Full Coverage: All samples will be processed

New Data Flow:
- Load full data (no sample limit)
- Process batch_size samples per iteration
- Generate rubrics for these samples
- Use MCR² to evaluate information gain
- Decide whether to continue based on gain
- Start new round after processing all samples

Parameter Description:
- batch_size: Number of samples processed per iteration, core control parameter
- model_name: Language model name (e.g., "qwen3-32b")
- max_workers: Number of concurrent threads, controls generation speed
- max_epochs: Maximum generation rounds per sample, controls single sample generation quality
- generate_number: Number of rubrics generated per sample
- mcr_batch_size: Number of rubrics selected by MCR each time
- min_increment_threshold: Minimum information gain threshold, determines stopping condition
- patience: Number of consecutive low increments to trigger stop, avoids random fluctuations
- max_iterations: Maximum iteration count, prevents infinite loops
- max_total_rubrics: Maximum total rubric count limit
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Core imports
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.rubric.generator import create_simple_generator

# Import main components
from rm_gallery.core.reward.rubric.mcr_selector import MCR2Config, MCR2Selector
from rm_gallery.core.utils.file import read_jsonl, write_json


class RubricMCRPipeline:
    """Simplified Rubric-MCR integrated pipeline"""

    def __init__(
        self,
        model_name: str,
        max_workers: int,
        max_epochs: int,
        batch_size: int,
        generate_number: int,
        mcr_batch_size: int,
        min_increment_threshold: float,
        patience: int,
        max_iterations: int,
        max_total_rubrics: int,
        min_success_rate: float = 0.3,
    ):
        """
        Initialize pipeline

        Args:
            model_name: Language model to use, affects generation quality
            max_workers: Number of concurrent threads, recommended 2-4x CPU cores
            max_epochs: Maximum generation rounds per sample, more rounds = higher quality but slower
            batch_size: Number of samples processed per iteration
            generate_number: Number of rubrics generated per sample
            mcr_batch_size: Number of rubrics selected by MCR each time
            min_increment_threshold: Information gain stopping threshold, smaller value = stricter convergence
            patience: Number of consecutive low increments to trigger stop, avoids random fluctuations
            max_iterations: Safety limit to prevent infinite iteration
            max_total_rubrics: Final output quantity limit
            min_success_rate: Minimum success rate threshold, early stop if below this value
        """
        # Save configuration parameters
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.generate_number = generate_number
        self.mcr_batch_size = mcr_batch_size
        self.min_increment_threshold = min_increment_threshold
        self.patience = patience
        self.max_iterations = max_iterations
        self.max_total_rubrics = max_total_rubrics
        self.min_success_rate = min_success_rate

        # Create config dictionary for compatibility
        self.config = {
            "model_name": model_name,
            "max_workers": max_workers,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "generate_number": generate_number,
            "mcr_batch_size": mcr_batch_size,
            "min_increment_threshold": min_increment_threshold,
            "patience": patience,
            "max_iterations": max_iterations,
            "max_total_rubrics": max_total_rubrics,
            "min_success_rate": min_success_rate,
        }

        # Core component initialization
        self.model = OpenaiLLM(model=model_name, enable_thinking=True)
        self.generator = create_simple_generator(llm=self.model, config=self.config)

        # Create MCR² selector with configuration
        self.mcr_config = MCR2Config(
            batch_size=mcr_batch_size,
            min_increment_threshold=0.0,  # No internal stopping, controlled externally
            patience=10,
            max_samples=max_total_rubrics,
            candidate_sample_ratio=0.3,
        )
        self.mcr_selector = MCR2Selector(config=self.mcr_config)

        # State tracking
        self.all_rubrics = []
        self.iteration_history = []
        self.coding_rates = []
        self.low_increment_count = 0  # Consecutive low increment counter
        self.current_sample_index = 0  # Current sample index being processed

    def transform(
        self, samples: List[dict], domains: List[str] = ["general"]
    ) -> DataSample:
        if domains:
            samples = [
                DataSample(**sample)
                for sample in samples
                if sample["metadata"]["domain"] in domains
            ]
        else:
            samples = [DataSample(**sample) for sample in samples]
        for sample in samples:
            for output in sample.output:
                output.answer.label["preference"] = (
                    "chosen" if output.answer.label["is_preferred"] else "rejected"
                )

        return samples

    def load_data(self, data_path: str) -> List[DataSample]:
        """Load and preprocess data (full load)"""
        logger.info(f"Loading data from {data_path}")
        raw_data = read_jsonl(data_path)
        samples = self.transform(raw_data, domains=None)  # No domain restriction

        logger.info(f"Loaded {len(samples)} samples for batch processing")
        return samples

    def generate_rubrics_batch(
        self, batch_samples: List[DataSample]
    ) -> Tuple[List[str], dict]:
        """Generate rubrics for a batch of samples and return statistics"""
        logger.info(f"Generating rubrics for batch of {len(batch_samples)} samples...")
        rubrics, processed_samples = self.generator.run_batch(
            batch_samples, max_workers=self.max_workers
        )

        # Filter valid rubrics
        valid_rubrics = [r for r in rubrics if r and len(r.strip()) > 10]

        # Calculate detailed statistics
        total_samples = len(processed_samples)
        successful_samples = sum(
            1
            for s in processed_samples
            if s.metadata.get("rubric_valid", "False") == "True"
        )
        failed_samples = total_samples - successful_samples
        success_rate = (
            successful_samples / total_samples * 100 if total_samples > 0 else 0
        )

        # Analyze failure reasons
        failed_epochs = {}
        for sample in processed_samples:
            if sample.metadata.get("rubric_valid", "False") == "False":
                epoch = sample.metadata.get("rubric_epoch", "unknown")
                failed_epochs[epoch] = failed_epochs.get(epoch, 0) + 1

        generation_stats = {
            "total_samples": total_samples,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "success_rate": success_rate,
            "failed_epochs": failed_epochs,
            "valid_rubrics": len(valid_rubrics),
        }

        logger.info(
            f"Generation completed: {successful_samples}/{total_samples} samples successful ({success_rate:.1f}%)"
        )
        logger.info(f"Generated {len(valid_rubrics)} valid rubrics")

        if failed_samples > 0:
            logger.warning(f"⚠️  {failed_samples} samples failed after max epochs")
            if failed_epochs:
                logger.info(f"Failure distribution by epoch: {failed_epochs}")

        return valid_rubrics, generation_stats

    def evaluate_mcr(self, new_rubrics: List[str]) -> Dict[str, Any]:
        """Evaluate information gain using MCR²"""
        combined = self.all_rubrics + new_rubrics
        if not combined:
            return {
                "selected_texts": [],
                "final_coding_rate": 0.0,
                "increment": 0.0,
                "final_sample_count": 0,
            }

        logger.info(
            f"📊 MCR² evaluation: {len(self.all_rubrics)} existing + {len(new_rubrics)} new = {len(combined)} total rubrics"
        )

        # MCR² selection with updated config
        selection_result = self.mcr_selector.select(
            texts=combined,
            max_samples=min(self.max_total_rubrics, len(combined)),
        )

        # Calculate gain
        previous_rate = self.coding_rates[-1] if self.coding_rates else 0.0
        current_rate = selection_result.final_coding_rate
        increment = current_rate - previous_rate

        # Return dict for backward compatibility
        results = {
            "selected_texts": selection_result.selected_texts,
            "final_coding_rate": selection_result.final_coding_rate,
            "increment": increment,
            "final_sample_count": selection_result.final_sample_count,
            "batch_history": selection_result.batch_history,
            "coding_rate_history": selection_result.coding_rate_history,
            "increment_history": selection_result.increment_history,
        }

        logger.info(
            f"📈 MCR² results: selected {selection_result.final_sample_count} rubrics, "
            f"coding_rate={current_rate:.6f}, increment={increment:.6f}"
        )
        return results

    def should_continue(
        self, mcr_results: Dict[str, Any], iteration: int, generation_stats: dict
    ) -> Tuple[bool, str]:
        """Determine whether to continue iteration - prioritize natural convergence conditions"""

        # 1. First check natural convergence conditions (these are main stopping reasons)

        # Check information gain - most important stopping condition
        increment = mcr_results.get("increment", 0.0)

        if increment < self.min_increment_threshold:
            self.low_increment_count += 1
            logger.info(
                f"Low increment detected: {increment:.6f} < {self.min_increment_threshold:.6f} "
                f"(count: {self.low_increment_count}/{self.patience})"
            )

            if self.low_increment_count >= self.patience:
                return (
                    False,
                    f"Converged: {self.patience} consecutive low increments (last: {increment:.6f} < {self.min_increment_threshold})",
                )
        else:
            # Reset counter
            if self.low_increment_count > 0:
                logger.info(
                    f"Increment recovered: {increment:.6f} >= {self.min_increment_threshold:.6f}, resetting counter"
                )
            self.low_increment_count = 0

        # Check success rate - quality issue
        success_rate = (
            generation_stats.get("success_rate", 0) / 100
        )  # Convert to 0-1 range
        if success_rate < self.min_success_rate:
            return (
                False,
                f"Quality issue: Success rate too low ({success_rate:.1%} < {self.min_success_rate:.1%})",
            )

        # 2. Then check resource limit conditions (these are protective measures)

        # Check quantity limit
        if len(self.all_rubrics) >= self.max_total_rubrics:
            return (
                False,
                f"Resource limit: Max rubrics reached ({self.max_total_rubrics})",
            )

        # Check iteration count - final safety net
        if iteration >= self.max_iterations:
            # Give reminder if there's still significant gain
            if increment > self.min_increment_threshold * 2:
                logger.warning(
                    f"⚠️ Stopping due to max iterations, but increment ({increment:.6f}) is still significant"
                )
                logger.warning(
                    "💡 Consider increasing max_iterations to allow further convergence"
                )
            return (
                False,
                f"Safety limit: Max iterations reached ({self.max_iterations})",
            )

        return True, ""

    def get_next_batch(
        self, all_samples: List[DataSample]
    ) -> Optional[List[DataSample]]:
        """Get next batch of samples"""
        start_idx = self.current_sample_index
        end_idx = min(start_idx + self.batch_size, len(all_samples))

        if start_idx >= len(all_samples):
            return None  # All data has been processed

        batch = all_samples[start_idx:end_idx]
        self.current_sample_index = end_idx

        logger.info(
            f"Getting batch: samples {start_idx}-{end_idx-1} ({len(batch)} samples)"
        )
        return batch

    def run(self, data_path: str) -> Dict[str, Any]:
        """Run complete pipeline - new batch processing logic"""
        logger.info("🚀 Starting Rubric-MCR Pipeline with batch processing...")
        start_time = time.time()

        # Load full data
        all_samples = self.load_data(data_path)
        logger.info(
            f"Will process {len(all_samples)} samples in batches of {self.batch_size}"
        )

        # Iterative batch processing
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"\n{'='*15} ITERATION {iteration} {'='*15}")

            # Get next batch
            batch_samples = self.get_next_batch(all_samples)
            if batch_samples is None:
                logger.info(
                    "🏁 All samples processed, checking if we should continue with new cycle..."
                )
                # Reset index, start new round (if there's still gain)
                self.current_sample_index = 0
                batch_samples = self.get_next_batch(all_samples)
                if batch_samples is None:
                    logger.error("Failed to get batch samples, stopping")
                    break

            # Generate rubrics for current batch
            new_rubrics, generation_stats = self.generate_rubrics_batch(batch_samples)
            if not new_rubrics:
                logger.warning(
                    "No valid rubrics generated for batch, continuing to next batch..."
                )
                continue

            # MCR² evaluation
            mcr_results = self.evaluate_mcr(new_rubrics)

            # Determine whether to continue
            should_continue, reason = self.should_continue(
                mcr_results, iteration, generation_stats
            )

            # Update state
            self.all_rubrics = mcr_results["selected_texts"]
            self.coding_rates.append(mcr_results["final_coding_rate"])
            self.iteration_history.append(
                {
                    "iteration": iteration,
                    "batch_start": self.current_sample_index - len(batch_samples),
                    "batch_end": self.current_sample_index - 1,
                    "batch_size": len(batch_samples),
                    "new_generated": len(new_rubrics),
                    "total_selected": len(self.all_rubrics),
                    "coding_rate": mcr_results["final_coding_rate"],
                    "increment": mcr_results["increment"],
                    "generation_stats": generation_stats,
                }
            )

            logger.info(
                f"Iteration {iteration}: batch[{self.current_sample_index - len(batch_samples)}:{self.current_sample_index-1}] → {len(new_rubrics)} new → {len(self.all_rubrics)} total rubrics"
            )
            logger.info(
                f"Coding rate: {mcr_results['final_coding_rate']:.6f} (+{mcr_results['increment']:.6f})"
            )

            if not should_continue:
                logger.info(f"🛑 Stopping: {reason}")
                break

        # Generate results
        total_time = time.time() - start_time

        results = {
            "config": self.config,
            "final_rubrics": self.all_rubrics,
            "total_iterations": iteration,
            "total_time": total_time,
            "iteration_history": self.iteration_history,
            "coding_rates": self.coding_rates,
            "final_coding_rate": self.coding_rates[-1] if self.coding_rates else 0.0,
        }

        logger.info(
            f"✅ Pipeline completed: {len(self.all_rubrics)} rubrics, {iteration} iterations, {total_time:.1f}s"
        )

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save results and visualization"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        write_json(results, str(output_path / "results.json"))
        write_json(results["final_rubrics"], str(output_path / "rubrics.json"))
        logger.info(f"Results saved to {output_path}")


def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Auto-Rubric Generation Pipeline")
    parser.add_argument(
        "--data-path",
        default="./exports/helpsteer3_train/helpsteer3_preference.jsonl",
        help="Path to preference dataset",
    )
    parser.add_argument("--model", default="qwen3-32b", help="Model name")
    parser.add_argument(
        "--output-base", default="./exports", help="Base output directory"
    )
    parser.add_argument("--max-workers", type=int, default=32, help="Max workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=1, help="Max epochs")
    parser.add_argument(
        "--generate-number", type=int, default=1, help="Generate number"
    )
    parser.add_argument("--mcr-batch-size", type=int, default=10, help="MCR batch size")
    parser.add_argument(
        "--min-increment-threshold",
        type=float,
        default=0.002,
        help="Min increment threshold",
    )
    parser.add_argument("--patience", type=int, default=2, help="Patience")
    parser.add_argument("--max-iterations", type=int, default=50, help="Max iterations")
    parser.add_argument(
        "--max-total-rubrics", type=int, default=200, help="Max total rubrics"
    )
    parser.add_argument(
        "--min-success-rate", type=float, default=0.3, help="Min success rate"
    )
    parser.add_argument(
        "--enable-structuring",
        type=bool,
        default=True,
        help="Enable Theme-Tips categorization",
    )
    parser.add_argument(
        "--num-categories", type=int, default=5, help="Number of Theme-Tips categories"
    )

    args = parser.parse_args()

    # Use parsed arguments
    model_name = args.model
    data_path = args.data_path
    max_workers = args.max_workers
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    generate_number = args.generate_number
    mcr_batch_size = args.mcr_batch_size
    min_increment_threshold = args.min_increment_threshold
    patience = args.patience
    max_iterations = args.max_iterations
    max_total_rubrics = args.max_total_rubrics
    min_success_rate = args.min_success_rate
    output_dir = f"{args.output_base}/{model_name}"

    try:
        pipeline = RubricMCRPipeline(
            model_name=model_name,
            max_workers=max_workers,
            max_epochs=max_epochs,
            batch_size=batch_size,
            generate_number=generate_number,
            mcr_batch_size=mcr_batch_size,
            min_increment_threshold=min_increment_threshold,
            patience=patience,
            max_iterations=max_iterations,
            max_total_rubrics=max_total_rubrics,
            min_success_rate=min_success_rate,
        )

        results = pipeline.run(data_path)

        # save results
        pipeline.save_results(results, output_dir)

        # Theme-Tips categorization
        if args.enable_structuring:
            logger.info("\n" + "=" * 60)
            logger.info("🎯 RUNNING THEME-TIPS CATEGORIZATION")
            logger.info("=" * 60)

            try:
                from rm_gallery.core.reward.rubric.structurer import RubricStructurer

                # initialize structurer
                structurer_output_dir = f"{output_dir}/structuring"
                structurer = RubricStructurer(
                    num_themes=args.num_categories,
                    model_name=model_name,
                    output_dir=structurer_output_dir,
                )

                # run structuring
                final_rubrics, themes = structurer.structure_rubrics(
                    results["final_rubrics"]
                )

                logger.info(
                    f"✅ Categorization completed: {len(final_rubrics)} Theme-Tips rubrics generated"
                )

                # print final Theme-Tips rubrics
                logger.info("\n📋 Final Theme-Tips Rubrics:")
                for i, rubric in enumerate(final_rubrics, 1):
                    lines = rubric.split("\n")
                    theme = lines[0] if lines else rubric[:100]
                    logger.info(f"  {i}. {theme}")
                    if len(lines) > 1:
                        logger.info(f"     ({len(lines)-1} tips)")

            except Exception as e:
                logger.error(f"❌ Categorization failed: {e}")
                logger.warning("Continuing without categorization...")

        # print summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Final results: {len(results['final_rubrics'])} raw rubrics")
        logger.info(f"Total iterations: {results['total_iterations']}")
        logger.info(f"Total time: {results['total_time']:.1f}s")
        logger.info(f"Final coding rate: {results['final_coding_rate']:.6f}")
        logger.info(f"Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
