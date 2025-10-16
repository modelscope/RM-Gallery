#!/usr/bin/env python3
"""
Query-Specific Rubric Generator

Core Ideas:
1. Query-Specific Generation: Generate rubrics specific to each query
2. Iterative Improvement: Improve rubrics iteratively
3. Evaluation: Evaluate rubrics using reward module
4. Stop Condition: Stop when rubrics converge or reach maximum epochs
5. Statistics: Collect statistics for analysis
"""

import copy
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from retry import retry
from tqdm import tqdm

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.message import format_messages
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.rubric.base import (
    RubricEvaluationTemplate,
    RubricGenerateTemplate,
    RubricReviseTemplate,
)


class RubricGenerator(BaseModel):
    """Simplified Rubric Generator - focused on generation, no clustering"""

    llm: OpenaiLLM = Field(default=..., description="Language model client")
    generate_number: int = Field(
        default=1, description="Number of rubrics to generate per sample"
    )
    max_retries: int = Field(default=5, description="Maximum retry attempts")
    max_workers: int = Field(default=32, description="Maximum concurrent threads")
    max_epochs: int = Field(default=5, description="Maximum iteration epochs")
    sample_timeout: int = Field(
        default=180, description="Maximum time (seconds) to process a single sample"
    )

    def generate_single(
        self, sample: DataSample, rubrics: Optional[List[str]] = None
    ) -> List[str]:
        """Generate rubrics for a single sample"""
        sample = copy.deepcopy(sample)
        query: str = format_messages(sample.input)

        # Process answers and preferences
        answers = [
            (output.answer.label["preference"], output.answer.content)
            for output in sample.output
        ]

        # Get evaluation reasoning (optional)
        critics = []
        if (
            "individual_preference" in sample.metadata
            and sample.metadata["individual_preference"]
        ):
            critics = [
                preference["reasoning"].replace("@Response", "@Answer")
                for preference in sample.metadata["individual_preference"]
                if "reasoning" in preference
            ]

        # Find the best answer
        best = None
        for i, (label, answer) in enumerate(answers):
            if label == "chosen":
                best = i + 1

        # Skip if no clear best answer
        if best is None:
            logger.warning("No clear best answer found, skipping sample")
            return []

        answers = [answer for _, answer in answers]

        # Generate prompt using RubricGenerateTemplate
        prompt = RubricGenerateTemplate.format(
            query=query,
            answers=answers,
            preference=best,
            critics=critics,
            number=self.generate_number,
            enable_thinking=self.llm.enable_thinking
            if hasattr(self.llm, "enable_thinking")
            else False,
        )

        # Call LLM for generation
        @retry(tries=self.max_retries, delay=1.0)
        def call_llm():
            response = self.llm.simple_chat(query=prompt)
            logger.debug(f"LLM response: {response}")
            result = RubricGenerateTemplate.parse(response)
            if len(result.rubrics) == 0:
                raise ValueError("No rubrics generated")
            return result.rubrics

        try:
            rubrics = call_llm()
            logger.debug(f"Generated {len(rubrics)} rubrics for sample")
            return rubrics
        except Exception as e:
            logger.error(f"Failed to generate rubrics: {str(e)}")
            return []

    def evaluate_single(self, sample: DataSample, rubrics: List[str]) -> DataSample:
        """Evaluate a single sample using the given rubrics"""
        try:
            # Format rubrics string
            rubrics_str = "\n".join(
                [f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics)]
            )

            # Get query and answers
            query = format_messages(sample.input)
            answers = [output.answer.content for output in sample.output]

            # Only support pairwise comparison
            if len(answers) != 2:
                raise ValueError(
                    "Evaluation only supports pairwise comparison (2 answers)"
                )

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

            # Convert to sample format
            evaluated_sample = copy.deepcopy(sample)
            preference = parsed.preference.upper()

            # Create reward scores based on preference
            scores = [0.0, 0.0]  # For 2 answers
            if preference == "A" or "RESPONSE A" in preference:
                scores = [1.0, 0.0]
            elif preference == "B" or "RESPONSE B" in preference:
                scores = [0.0, 1.0]
            elif preference == "TIE" or "EQUAL" in preference:
                scores = [0.5, 0.5]

            # Update sample metadata
            if not hasattr(evaluated_sample, "metadata"):
                evaluated_sample.metadata = {}

            evaluated_sample.metadata["reward_score"] = scores
            evaluated_sample.metadata["preference"] = preference

            # CRITICAL FIX: Also update the reward scores in output.answer.reward.score
            # This is needed for _check_sample_correctness to work properly
            for i, output in enumerate(evaluated_sample.output):
                if i < len(scores):
                    output.answer.reward.score = scores[i]

            logger.debug(f"Rubric evaluation: preference={preference}, scores={scores}")

            return evaluated_sample

        except Exception as e:
            logger.error(f"Failed to evaluate sample: {str(e)}")
            return sample

    def _check_sample_correctness(self, sample: DataSample) -> bool:
        """
        Check if sample is correct: whether the answer with highest reward score is "chosen"

        This is consistent with _split_samples logic:
        - True: highest score answer is "chosen", evaluation successful
        - False: highest score answer is not "chosen", evaluation failed
        """
        try:
            # Get reward scores for all answers
            reward_scores = [output.answer.reward.score for output in sample.output]

            # Find index of highest score answer
            max_idx = np.argmax(reward_scores)

            # Check if highest score answer is "chosen"
            is_chosen = sample.output[max_idx].answer.label["preference"] == "chosen"

            logger.debug(
                f"Reward scores: {reward_scores}, max_idx: {max_idx}, is_chosen: {is_chosen}"
            )
            return is_chosen

        except Exception as e:
            logger.error(f"Error checking sample correctness: {e}")
            return False

    def revise_rubrics(self, sample: DataSample, rubrics: List[str]) -> List[str]:
        """Revise rubrics based on evaluation results"""
        sample = copy.deepcopy(sample)
        query: str = format_messages(sample.input)

        # Process answers and preferences
        answers = [
            (output.answer.label["preference"], output.answer.content)
            for output in sample.output
        ]

        # Get evaluation reasoning (optional)
        critics = []
        if (
            "individual_preference" in sample.metadata
            and sample.metadata["individual_preference"]
        ):
            critics = [
                preference["reasoning"].replace("@Response", "@Answer")
                for preference in sample.metadata["individual_preference"]
                if "reasoning" in preference
            ]

        # Find the best answer
        best = None
        for i, (label, answer) in enumerate(answers):
            if label == "chosen":
                best = i + 1

        # Skip if no clear best answer
        if best is None:
            logger.warning("No clear best answer found for revision, skipping")
            return []

        answers = [answer for _, answer in answers]

        # Use RubricReviseTemplate to generate revision prompt
        prompt = RubricReviseTemplate.format(
            query=query,
            answers=answers,
            preference=best,
            critics=critics,
            number=self.generate_number,
            rubrics=rubrics,  # Pass previous rubrics for reference
            enable_thinking=self.llm.enable_thinking
            if hasattr(self.llm, "enable_thinking")
            else False,
        )

        # Call LLM for revision
        @retry(tries=self.max_retries, delay=1.0)
        def call_llm():
            response = self.llm.simple_chat(query=prompt)
            logger.debug(f"LLM revision response: {response}")
            result = RubricReviseTemplate.parse(response)
            if len(result.rubrics) == 0:
                raise ValueError("No revised rubrics generated")
            return result.rubrics

        try:
            revised_rubrics = call_llm()
            logger.debug(
                f"Revised {len(revised_rubrics)} rubrics based on previous {len(rubrics)} rubrics"
            )
            return revised_rubrics
        except Exception as e:
            logger.error(f"Failed to revise rubrics: {str(e)}")
            # Return empty to stop iteration (don't fallback to avoid repeated failures)
            return []

    def generate_iterative_single(
        self, sample: DataSample, progress_callback=None
    ) -> DataSample:
        """Perform iterative generation and improvement for a single sample

        Args:
            sample: Input sample to process
            progress_callback: Optional callback function(epoch, max_epochs) to report progress
        """
        sample = copy.deepcopy(sample)

        # Initial generation
        if progress_callback:
            progress_callback(0, self.max_epochs, "Generating...")

        rubrics = self.generate_single(sample)

        # Check if initial generation succeeded
        if not rubrics:
            logger.debug(
                "Initial generation failed (no clear best answer or generation error)"
            )
            sample.metadata["rubrics"] = []
            sample.metadata["rubric_valid"] = "False"
            sample.metadata["rubric_epoch"] = "0"
            return sample

        # Iterative improvement
        last_epoch = 0
        for epoch in range(self.max_epochs):
            last_epoch = epoch + 1  # Track the actual epoch number (1-indexed)

            # Report progress
            if progress_callback:
                progress_callback(epoch + 1, self.max_epochs, "Evaluating...")

            # Evaluate current rubrics
            evaluated_sample = self.evaluate_single(sample, rubrics)

            # logger.info(f"Evaluated sample: {evaluated_sample}")

            # Check if evaluation passes
            # Use same logic as _split_samples: check if highest score answer is "chosen"
            is_correct = self._check_sample_correctness(evaluated_sample)

            # Enhanced debugging
            try:
                reward_scores = [
                    output.answer.reward.score for output in evaluated_sample.output
                ]
                preferences = [
                    output.answer.label["preference"]
                    for output in evaluated_sample.output
                ]
                logger.debug(
                    f"Epoch {epoch+1}: reward_scores={reward_scores}, preferences={preferences}, is_correct={is_correct}"
                )
            except Exception as debug_e:
                logger.debug(
                    f"Epoch {epoch+1}: sample correctness = {is_correct}, debug_error={debug_e}"
                )

            if is_correct:
                # Evaluation passed, stop iteration
                sample.metadata["rubrics"] = rubrics
                sample.metadata["rubric_valid"] = "True"
                sample.metadata["rubric_epoch"] = str(last_epoch)
                logger.debug(f"Sample converged at epoch {last_epoch}")
                return sample

            # Evaluation failed, try to improve
            if progress_callback:
                progress_callback(epoch + 1, self.max_epochs, "Revising...")

            revised_rubrics = self.revise_rubrics(evaluated_sample, rubrics)
            if not revised_rubrics:
                # Revise failed, mark the epoch where it stopped
                logger.debug(f"Revise failed at epoch {last_epoch}, stopping iteration")
                break

            rubrics = revised_rubrics
            # logger.debug(f"Epoch {epoch+1}: revised rubrics")  # Commented out, too verbose

        # Iteration ended (not converged)
        sample.metadata["rubrics"] = rubrics
        sample.metadata["rubric_valid"] = "False"
        sample.metadata["rubric_epoch"] = str(
            last_epoch
        )  # Record actual last epoch, not max_epochs

        return sample

    def run_batch(
        self, samples: List[DataSample], max_workers: Optional[int] = None
    ) -> Tuple[List[str], List[DataSample]]:
        """Process samples in batch with timeout support"""
        logger.info(f"Processing {len(samples)} samples in batch")
        logger.info(f"Sample timeout: {self.sample_timeout}s")

        # Parameter processing
        max_workers = max_workers or self.max_workers

        # Track current progress for each sample
        sample_progress = {}

        # Parallel processing (with progress bar)
        processed_samples = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create progress callback for each sample
            def make_progress_callback(sample_idx):
                def callback(epoch, max_epochs, stage):
                    sample_progress[sample_idx] = {
                        "epoch": epoch,
                        "max_epochs": max_epochs,
                        "stage": stage,
                        "last_update": time.time(),
                    }

                return callback

            # Submit all tasks with progress callbacks
            futures = {
                executor.submit(
                    self.generate_iterative_single, sample, make_progress_callback(i)
                ): (i, sample)
                for i, sample in enumerate(samples)
            }

            # Use progress bar to show completion status
            with tqdm(
                total=len(samples), desc="Processing samples", unit="sample"
            ) as pbar:
                for future in as_completed(futures, timeout=None):
                    try:
                        # Try to get result with timeout
                        result = future.result(timeout=self.sample_timeout)
                        sample_idx, sample = futures[future]
                        processed_samples.append(
                            (sample_idx, result)
                        )  # Save index and result

                        # Update progress bar with detailed status
                        valid = result.metadata.get("rubric_valid", "False") == "True"
                        rubrics_count = len(result.metadata.get("rubrics", []))
                        epoch = result.metadata.get("rubric_epoch", "?")

                        # Get current active sample info
                        # Use list() to avoid "dictionary changed size during iteration" error
                        active_samples = [
                            idx
                            for idx in list(sample_progress.keys())
                            if idx not in [s[0] for s in processed_samples]
                        ]

                        if active_samples:
                            # Show info of one active sample
                            # Use try-except to handle potential concurrent modification
                            try:
                                active_idx = active_samples[0]
                                progress_info = sample_progress.get(active_idx, {})
                                current_epoch = progress_info.get("epoch", 0)
                                max_epochs = progress_info.get(
                                    "max_epochs", self.max_epochs
                                )
                                stage = progress_info.get("stage", "...")

                                status = (
                                    f"✓{rubrics_count}r@E{epoch}"
                                    if valid
                                    else f"✗@E{epoch}"
                                )
                                pbar.set_postfix_str(
                                    f"Last: {status} | Active: E{current_epoch}/{max_epochs} {stage}"
                                )
                            except (KeyError, RuntimeError):
                                # Handle race condition if dict changes during access
                                status = (
                                    f"✓{rubrics_count}r@E{epoch}"
                                    if valid
                                    else f"✗@E{epoch}"
                                )
                                pbar.set_postfix_str(f"Last: {status}")
                        else:
                            status = (
                                f"✓{rubrics_count}r@E{epoch}"
                                if valid
                                else f"✗@E{epoch}"
                            )
                            pbar.set_postfix_str(f"Last: {status}")

                        pbar.update(1)

                    except TimeoutError:
                        sample_idx, sample = futures[future]
                        logger.warning(
                            f"Sample {sample_idx} timed out after {self.sample_timeout}s"
                        )
                        # Create a failed sample
                        timeout_sample = copy.deepcopy(sample)
                        timeout_sample.metadata["rubrics"] = []
                        timeout_sample.metadata["rubric_valid"] = "False"
                        timeout_sample.metadata["rubric_epoch"] = "timeout"
                        timeout_sample.metadata["timeout"] = True
                        processed_samples.append((sample_idx, timeout_sample))
                        pbar.update(1)

                    except Exception as e:
                        sample_idx, sample = futures[future]
                        logger.error(f"Sample {sample_idx} processing failed: {e}")
                        # Create a failed sample
                        failed_sample = copy.deepcopy(sample)
                        failed_sample.metadata["rubrics"] = []
                        failed_sample.metadata["rubric_valid"] = "False"
                        failed_sample.metadata["rubric_epoch"] = "error"
                        failed_sample.metadata["error"] = str(e)
                        processed_samples.append((sample_idx, failed_sample))
                        pbar.update(1)

            # Sort results by original order
            processed_samples.sort(key=lambda x: x[0])
            processed_samples = [result for _, result in processed_samples]

        # Separate successful and failed samples
        successful_samples = []
        failed_samples = []
        all_rubrics = []

        for sample in processed_samples:
            is_valid = sample.metadata.get("rubric_valid", "False") == "True"
            sample_rubrics = sample.metadata.get("rubrics", [])

            if is_valid and sample_rubrics:
                successful_samples.append(sample)
                all_rubrics.extend(sample_rubrics)
            else:
                failed_samples.append(sample)

        # Statistics (simplified output)
        success_rate = (
            len(successful_samples) / len(processed_samples) * 100
            if processed_samples
            else 0
        )
        logger.info(
            f"Batch completed: {len(successful_samples)}/{len(processed_samples)} successful ({success_rate:.1f}%), {len(all_rubrics)} rubrics generated"
        )

        if failed_samples:
            logger.warning(f"{len(failed_samples)} samples failed")
            # Count timeout vs other failures
            timeout_count = sum(
                1 for s in failed_samples if s.metadata.get("timeout", False)
            )
            error_count = sum(1 for s in failed_samples if s.metadata.get("error"))
            other_count = len(failed_samples) - timeout_count - error_count

            if timeout_count:
                logger.warning(
                    f"  - {timeout_count} samples timed out (>{self.sample_timeout}s)"
                )
            if error_count:
                logger.warning(f"  - {error_count} samples had errors")
            if other_count:
                logger.warning(f"  - {other_count} samples failed to converge")

            # Failure details only shown in debug mode
            for i, sample in enumerate(
                failed_samples[:3]
            ):  # Only log first 3 failed samples
                epoch = sample.metadata.get("rubric_epoch", "unknown")
                logger.debug(f"Failed sample {i+1}: stopped at epoch {epoch}")

        return all_rubrics, processed_samples


def create_simple_generator(llm: OpenaiLLM, config: dict) -> RubricGenerator:
    """Create simplified generator instance"""
    return RubricGenerator(
        llm=llm,
        generate_number=config.get("generate_number", 1),
        max_retries=config.get("max_retries", 5),
        max_workers=config.get("max_workers", 32),
        max_epochs=config.get("max_epochs", 5),
        sample_timeout=config.get("sample_timeout", 180),
    )
