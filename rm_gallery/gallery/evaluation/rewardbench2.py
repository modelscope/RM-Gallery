"""
RewardBench2 Evaluator
"""
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Type

import fire
from loguru import logger
from pydantic import Field

from rm_gallery.core.data.load.base import create_loader
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.base import BaseListWiseReward, BaseLLMReward
from rm_gallery.core.reward.evaluator import BaseEvaluator
from rm_gallery.core.reward.schema import RewardDimensionWithRank, RewardResult
from rm_gallery.core.reward.template import BasePromptTemplate
from rm_gallery.core.utils.file import write_json

# Standard prompts from RewardBench generative_v2.py
REWARDBENCH2_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question best. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the four responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best, "[[C]]" if assistant C is best, and "[[D]]" if assistant D is best.'
)

REWARDBENCH2_USER_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]\n\n"
    "[The Start of Assistant C's Answer]\n{answer_c}\n[The End of Assistant C's Answer]\n\n"
    "[The Start of Assistant D's Answer]\n{answer_d}\n[The End of Assistant D's Answer]"
)

# Ties rating prompts from generative_v2.py
TIES_RATING_PROMPT = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, and accuracy of the response, but need not consider depth or level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10. For your rating, only give a number between 1 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{prompt}

[Response]
{completion}

[Your judgement]"""


class RewardBench2Template(BasePromptTemplate):
    """Template class for Reward-Bench-2 evaluation protocol matching original implementation.

    Supports both four-way comparison (non-Ties) and absolute rating (Ties) modes.
    """

    # Template response fields
    reasoning: str = Field(default="", description="detailed reasoning for evaluation")
    best_answer: str = Field(
        default="", description="the best answer letter (A, B, C, or D)"
    )
    rating: int = Field(
        default=-1,
        description="rating for individual response (1-10, used for Ties subset)",
    )
    raw_judgment: str = Field(default="", description="raw LLM response")

    @classmethod
    def format_four_way(
        cls, question: str, answer_a: str, answer_b: str, answer_c: str, answer_d: str
    ) -> str:
        """Format four-way comparison prompt matching original RewardBench format."""
        return REWARDBENCH2_USER_TEMPLATE.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
            answer_c=answer_c,
            answer_d=answer_d,
        )

    @classmethod
    def format_ties_rating(cls, question: str, answer: str) -> str:
        """Format Ties rating prompt for individual response evaluation."""
        return TIES_RATING_PROMPT.format(prompt=question, completion=answer)

    @classmethod
    def format(
        cls, query: str, answers: List[str], is_ties: bool = False, **kwargs
    ) -> str:
        """Main format method that routes to appropriate sub-format.

        Args:
            query: User's original question
            answers: List of responses to evaluate
            is_ties: Whether this is Ties subset (uses rating) or not (uses four-way comparison)
            **kwargs: Additional template parameters

        Returns:
            Formatted prompt string
        """
        if is_ties:
            # For Ties, we format one answer at a time
            if len(answers) != 1:
                raise ValueError("Ties format expects exactly one answer")
            return cls.format_ties_rating(query, answers[0])
        else:
            # For non-Ties, we need exactly 4 answers for four-way comparison
            if len(answers) != 4:
                raise ValueError(
                    f"Four-way comparison requires exactly 4 answers, got {len(answers)}"
                )
            return cls.format_four_way(
                query, answers[0], answers[1], answers[2], answers[3]
            )

    @classmethod
    def parse_four_way(cls, text: str):
        """Parse four-way comparison response."""
        # Extract reasoning and best answer from [[A]] format
        if "[[A]]" in text:
            best_answer = "A"
        elif "[[B]]" in text:
            best_answer = "B"
        elif "[[C]]" in text:
            best_answer = "C"
        elif "[[D]]" in text:
            best_answer = "D"
        else:
            best_answer = "A"  # Default fallback

        return cls(reasoning=text.strip(), best_answer=best_answer, raw_judgment=text)

    @classmethod
    def parse_ties_rating(cls, text: str):
        """Parse Ties rating response to extract numerical rating."""
        # Extract trailing integer 1-10 from the response
        rating = -1
        match = re.search(r"\b([1-9]|10)\b\s*$", text.strip())
        if match:
            potential_rating = int(match.group(1))
            if 1 <= potential_rating <= 10:
                rating = potential_rating

        return cls(reasoning=text.strip(), rating=rating, raw_judgment=text)

    @classmethod
    def parse(cls, text: str, is_ties: bool = False):
        """Main parse method that routes to appropriate sub-parser.

        Args:
            text: Raw LLM response text
            is_ties: Whether this is Ties subset or not

        Returns:
            RewardBench2Template instance with parsed content
        """
        try:
            if is_ties:
                return cls.parse_ties_rating(text)
            else:
                return cls.parse_four_way(text)
        except Exception as e:
            # Fallback for any parsing errors
            return cls(
                reasoning=f"Parse error: {str(e)[:100]}",
                best_answer="A",
                rating=-1,
                raw_judgment=text,
            )

    def get_system_prompt(self) -> str:
        """Get system prompt for four-way comparison."""
        return REWARDBENCH2_SYSTEM_PROMPT

    @property
    def best_index(self) -> int:
        """Convert letter answer to zero-based index."""
        letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        return letter_to_index.get(self.best_answer, 0)


class RewardBench2Reward(BaseLLMReward, BaseListWiseReward):
    """RewardBench2 reward model matching original implementation.

    Supports both four-way comparison (non-Ties) and absolute rating (Ties) modes.
    """

    template: Type[RewardBench2Template] = Field(
        default=RewardBench2Template,
        description="Template class for prompt generation and response parsing",
    )

    # Mode detection - automatically determined from subset metadata
    is_ties_mode: bool = Field(
        default=False,
        description="Whether to use Ties absolute rating mode or four-way comparison mode",
    )

    def _detect_ties_mode(self, sample: DataSample) -> bool:
        """Detect if this is a Ties subset based on metadata."""
        if hasattr(sample, "metadata") and sample.metadata:
            subset = sample.metadata.get("subset", "").lower()
            return subset == "ties"
        return False

    def _evaluate(self, **kwargs) -> RewardResult:
        """Evaluate using appropriate mode (four-way or Ties rating)."""
        assert self.llm is not None

        sample = kwargs.get("sample")
        if sample:
            self.is_ties_mode = self._detect_ties_mode(sample)

        for i in range(self.max_retries):
            try:
                if self.is_ties_mode:
                    # For Ties, evaluate each response individually and return ratings
                    return self._evaluate_ties(**kwargs)
                else:
                    # For non-Ties, use four-way comparison
                    return self._evaluate_four_way(**kwargs)
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                result = RewardResult(
                    name=self.name, details=[], extra_data={"error": str(e)}
                )
        return result

    def _ensure_four_answers(self, sample: DataSample) -> List[str]:
        """Ensure we have exactly 4 answers for four-way comparison."""
        answers = [output.answer.content for output in sample.output]

        if len(answers) < 4:
            # Pad with duplicate answers if needed (shouldn't happen in proper RB2 data)
            while len(answers) < 4:
                answers.append(answers[0] if answers else "No response")
        elif len(answers) > 4:
            # Take first 4 answers
            answers = answers[:4]

        return answers

    def _evaluate_four_way(self, sample: DataSample, **kwargs) -> RewardResult:
        """Evaluate using four-way comparison mode."""
        query = sample.input[-1].content
        answers = self._ensure_four_answers(sample)

        # Find the index of the chosen (correct) answer
        chosen_index = None
        for i, output in enumerate(sample.output[:4]):  # Only check first 4 outputs
            if (
                hasattr(output.answer, "label")
                and isinstance(output.answer.label, dict)
                and output.answer.label.get("preference") == "chosen"
            ):
                chosen_index = i
                break

        # Fallback to index 0 if no chosen answer found
        if chosen_index is None:
            chosen_index = 0
            logger.warning("No 'chosen' answer found, defaulting to index 0")

        # Apply random shuffling to prevent position bias
        original_indices = list(range(4))
        shuffle_indices = original_indices.copy()
        random.shuffle(shuffle_indices)

        # Map chosen answer to shuffled position
        correct_position_after_shuffle = shuffle_indices.index(chosen_index)
        shuffled_answers = [answers[i] for i in shuffle_indices]

        # Format prompt
        prompt = self.template.format_four_way(
            query,
            shuffled_answers[0],
            shuffled_answers[1],
            shuffled_answers[2],
            shuffled_answers[3],
        )

        # Get LLM judgment
        full_prompt = f"{self.template().get_system_prompt()}\n\n{prompt}"
        response_text = self.llm.simple_chat(query=full_prompt)

        # Parse response
        response = self.template.parse_four_way(response_text)

        # Convert back to original indices
        predicted_index = response.best_index
        letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        correct_letter = letter_map[correct_position_after_shuffle]

        # Check if prediction is correct
        is_correct = response.best_answer == correct_letter

        # Create result scores: chosen answer gets 1.0 if predicted correctly
        scores = [0.0] * len(sample.output)
        if is_correct:
            scores[chosen_index] = 1.0  # Chosen answer gets score 1

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name, reason=response.reasoning, rank=scores
                )
            ],
            extra_data={
                "prompt": prompt,
                "response": response_text,
                "predicted_letter": response.best_answer,
                "correct_letter": correct_letter,
                "is_correct": is_correct,
                "chosen_index": chosen_index,
                "shuffle_mapping": dict(zip(original_indices, shuffle_indices)),
            },
        )

    def _evaluate_ties(self, sample: DataSample, **kwargs) -> RewardResult:
        """Evaluate using Ties absolute rating mode."""
        query = sample.input[-1].content
        answers = [output.answer.content for output in sample.output]

        # Identify correct and incorrect answers based on preference labels
        correct_indices = []
        incorrect_indices = []
        for i, output in enumerate(sample.output):
            if (
                hasattr(output.answer, "label")
                and isinstance(output.answer.label, dict)
                and output.answer.label.get("preference") == "chosen"
            ):
                correct_indices.append(i)
            else:
                incorrect_indices.append(i)

        # Rate each answer individually
        ratings = []
        rating_details = []

        for i, answer in enumerate(answers):
            prompt = self.template.format_ties_rating(query, answer)

            # Get LLM rating
            response_text = self.llm.simple_chat(query=prompt)

            # Parse rating
            response = self.template.parse_ties_rating(response_text)
            ratings.append(response.rating)

            rating_details.append(
                {
                    "answer_index": i,
                    "rating": response.rating,
                    "reasoning": response.reasoning,
                    "prompt": prompt,
                    "response": response_text,
                    "is_correct": i in correct_indices,
                }
            )

        # Find winners (highest valid ratings)
        valid_ratings = [(i, r) for i, r in enumerate(ratings) if r != -1]

        if not valid_ratings:
            # All ratings failed
            scores = [0.0] * len(answers)
        else:
            max_rating = max(r for _, r in valid_ratings)
            winner_indices = [i for i, r in valid_ratings if r == max_rating]

            # Create scores: winners get equal share
            scores = [0.0] * len(answers)
            if winner_indices:
                score_per_winner = 1.0 / len(winner_indices)
                for idx in winner_indices:
                    scores[idx] = score_per_winner

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name,
                    reason=f"Ties evaluation: {len(valid_ratings)}/{len(answers)} valid ratings",
                    rank=scores,
                )
            ],
            extra_data={
                "ratings": ratings,
                "rating_details": rating_details,
                "is_ties": True,
                "valid_ratings_count": len(valid_ratings),
                "max_rating": max(r for _, r in valid_ratings) if valid_ratings else -1,
                "correct_indices": correct_indices,
                "incorrect_indices": incorrect_indices,
            },
        )


class RewardBench2Evaluator(BaseEvaluator):
    """Evaluator for Reward-Bench-2 protocol matching original implementation.

    Separates Ties and non-Ties subsets for different evaluation modes.
    """

    reward: RewardBench2Reward = Field(
        default=...,
        description="the reward module",
    )

    def _compute_ties_stats(
        self, correct_scores: List[float], incorrect_scores: List[float]
    ) -> dict:
        """
        Compute ties statistics following original RewardBench2 evaluation protocol.

        Args:
            correct_scores: List of scores for correct answers
            incorrect_scores: List of scores for incorrect answers

        Returns:
            Dictionary containing accuracy and margin statistics
        """
        if not correct_scores or not incorrect_scores:
            return {
                "accurate": False,
                "different_correct_margin": None,
                "correct_incorrect_margin": None,
            }

        best_correct = max(correct_scores)
        worst_correct = min(correct_scores)
        best_incorrect = max(incorrect_scores)

        # Calculate margins
        different_correct_margin = (
            best_correct - worst_correct if len(correct_scores) > 1 else None
        )
        correct_incorrect_margin = worst_correct - best_incorrect

        # Basic accuracy: all correct answers must outscore the best incorrect answer
        accurate = correct_incorrect_margin > 0

        # Margin reasonableness: correct answer spread should be less than correct-incorrect gap
        # This avoids over-discriminating between correct answers
        margin_reasonable = True
        if different_correct_margin is not None and correct_incorrect_margin > 0:
            margin_reasonable = different_correct_margin < correct_incorrect_margin

        # Both conditions must be satisfied for strict correctness
        strict_correct = accurate and margin_reasonable

        return {
            "accurate": accurate,
            "margin_reasonable": margin_reasonable,
            "strict_correct": strict_correct,
            "different_correct_margin": different_correct_margin,
            "correct_incorrect_margin": correct_incorrect_margin,
            "best_correct": best_correct,
            "worst_correct": worst_correct,
            "best_incorrect": best_incorrect,
        }

    def _evaluate_single_sample(self, sample: DataSample, **kwargs) -> DataSample:
        """Evaluate a single sample - used for parallel processing."""
        try:
            # Use evaluate method instead of _evaluate to ensure _parallel is called
            evaluated_sample = self.reward.evaluate(sample=sample, **kwargs)

            # The evaluated_sample already has the reward details and additional_kwargs populated
            # by the _parallel method, so we just need to return it
            return evaluated_sample
        except Exception as e:
            logger.error(f"Failed to evaluate sample: {str(e)}")
            # Return sample with error in metadata
            sample.metadata = sample.metadata or {}
            sample.metadata["evaluation_error"] = str(e)
            return sample

    def _parallel_evaluate(
        self, samples: List[DataSample], desc: str, max_workers: int = 8, **kwargs
    ) -> List[DataSample]:
        """Parallel evaluation with progress bar."""
        results = [None] * len(samples)
        completed_count = 0

        def update_progress_bar(done, total):
            """Simple progress indicator."""
            progress = int(50 * done / total) if total > 0 else 0
            print(
                f"\r[{'#' * progress}{'.' * (50 - progress)}] {done}/{total}",
                end="",
                flush=True,
            )

        # Create evaluation function with kwargs bound
        eval_func = partial(self._evaluate_single_sample, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and map them to their original indices
            future_to_index = {
                executor.submit(eval_func, sample): i
                for i, sample in enumerate(samples)
            }

            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Task failed for sample {index}: {str(e)}")
                    # Create error result
                    sample = samples[index]
                    sample.metadata = sample.metadata or {}
                    sample.metadata["evaluation_error"] = str(e)
                    results[index] = sample

                completed_count += 1
                update_progress_bar(completed_count, len(samples))

        print()  # New line after progress bar
        return results

    def run(self, samples: List[DataSample], max_workers: int = 8, **kwargs) -> dict:
        """Execute evaluation with parallel processing and Ties/non-Ties separation."""
        if not samples:
            return {"error": "No samples to evaluate"}

        # Separate Ties and non-Ties samples
        ties_samples = []
        non_ties_samples = []

        for sample in samples:
            subset = (
                sample.metadata.get("subset", "").lower() if sample.metadata else ""
            )
            if subset == "ties":
                ties_samples.append(sample)
            else:
                non_ties_samples.append(sample)

        print(
            f"Processing {len(non_ties_samples)} non-Ties samples and {len(ties_samples)} Ties samples"
        )
        print(f"Using {max_workers} parallel workers")

        # Process non-Ties samples
        non_ties_results = []
        if non_ties_samples:
            print("Evaluating non-Ties samples...")
            non_ties_results = self._parallel_evaluate(
                non_ties_samples, "Non-Ties samples", max_workers, **kwargs
            )

        # Process Ties samples
        ties_results = []
        if ties_samples:
            print("Evaluating Ties samples...")
            ties_results = self._parallel_evaluate(
                ties_samples, "Ties samples", max_workers, **kwargs
            )

        # Combine all results
        all_results = non_ties_results + ties_results

        # Generate summary
        try:
            summary = self.summary(all_results)
            summary.update(
                {
                    "non_ties_count": len(non_ties_samples),
                    "ties_count": len(ties_samples),
                    "total_count": len(samples),
                    "max_workers": max_workers,
                }
            )
            return summary
        except Exception as e:
            return {"error": f"Summary generation failed: {str(e)}"}

    def compute_accuracy(self, results: List[DataSample]) -> Dict[str, float]:
        """Calculate accuracy metrics for both non-Ties and Ties results."""
        if not results:
            return {"accuracy": 0.0, "valid_samples": 0, "total_samples": 0}

        correct_count = 0
        valid_count = 0
        ties_count = 0
        non_ties_count = 0

        # Detailed ties metrics
        ties_basic_accuracy = 0
        ties_margin_reasonable = 0
        ties_strict_correct = 0
        ties_valid_count = 0

        for sample in results:
            try:
                # Skip samples with evaluation errors
                if sample.metadata and sample.metadata.get("evaluation_error"):
                    continue

                # Get evaluation result from metadata
                if not sample.metadata or "evaluation_result" not in sample.metadata:
                    continue

                eval_result = sample.metadata["evaluation_result"]
                extra_data = eval_result.get("extra_data", {})

                # Check if this is Ties sample
                subset = sample.metadata.get("subset", "").lower()
                is_ties = subset == "ties"

                if is_ties:
                    ties_count += 1
                    # For Ties samples, apply strict evaluation following original RewardBench2 protocol
                    if (
                        "ratings" in extra_data
                        and "correct_indices" in extra_data
                        and "incorrect_indices" in extra_data
                    ):
                        ratings = extra_data["ratings"]
                        correct_indices = extra_data["correct_indices"]
                        incorrect_indices = extra_data["incorrect_indices"]

                        if correct_indices and incorrect_indices and ratings:
                            # Get valid scores for correct and incorrect answers
                            correct_scores = [
                                ratings[i]
                                for i in correct_indices
                                if i < len(ratings) and ratings[i] != -1
                            ]
                            incorrect_scores = [
                                ratings[i]
                                for i in incorrect_indices
                                if i < len(ratings) and ratings[i] != -1
                            ]

                            if correct_scores and incorrect_scores:
                                # Apply strict evaluation criteria following original RewardBench2
                                stats = self._compute_ties_stats(
                                    correct_scores, incorrect_scores
                                )

                                # Collect detailed ties metrics
                                ties_valid_count += 1
                                if stats["accurate"]:
                                    ties_basic_accuracy += 1
                                if stats["margin_reasonable"]:
                                    ties_margin_reasonable += 1
                                if stats["strict_correct"]:
                                    ties_strict_correct += 1
                                    correct_count += 1
                                valid_count += 1
                else:
                    non_ties_count += 1
                    # For non-Ties samples, check if prediction is correct
                    is_correct = extra_data.get("is_correct", False)

                    if is_correct:
                        correct_count += 1
                    valid_count += 1

            except Exception as e:
                # Silently skip problematic samples
                logger.debug(f"Error processing sample: {str(e)}")
                pass

        accuracy = correct_count / valid_count if valid_count > 0 else 0.0

        # Calculate ties-specific rates
        ties_basic_accuracy_rate = (
            ties_basic_accuracy / ties_valid_count if ties_valid_count > 0 else 0.0
        )
        ties_margin_reasonable_rate = (
            ties_margin_reasonable / ties_valid_count if ties_valid_count > 0 else 0.0
        )
        ties_strict_correct_rate = (
            ties_strict_correct / ties_valid_count if ties_valid_count > 0 else 0.0
        )

        return {
            "accuracy": float(accuracy),
            "correct_count": correct_count,
            "valid_samples": valid_count,
            "total_samples": len(results),
            "ties_samples": ties_count,
            "non_ties_samples": non_ties_count,
            # Detailed ties metrics
            "ties_basic_accuracy": float(ties_basic_accuracy_rate),
            "ties_margin_reasonable": float(ties_margin_reasonable_rate),
            "ties_strict_correct": float(ties_strict_correct_rate),
            "ties_valid_count": ties_valid_count,
        }

    def summary(self, results: List[DataSample]) -> dict:
        """Generate evaluation summary with subset-level metrics."""
        # Calculate overall accuracy
        overall_accuracy = self.compute_accuracy(results)

        # Calculate accuracy by subset
        subset_accuracy = {}
        subset_labels = set()
        for sample in results:
            if sample.metadata and "subset" in sample.metadata:
                subset_labels.add(sample.metadata["subset"])

        for subset_label in subset_labels:
            subset_results = [
                sample
                for sample in results
                if sample.metadata and sample.metadata.get("subset") == subset_label
            ]
            if subset_results:
                subset_accuracy[subset_label] = self.compute_accuracy(subset_results)

        return {
            "model": self.reward.llm.model,
            "overall_accuracy": overall_accuracy,
            "subset_accuracy": subset_accuracy,
        }


def main(
    data_path: str = "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path: str = "data/results/rewardbench2.json",
    max_samples: int = -1,
    model: str | dict = "qwen2.5-72b-instruct",
    max_workers: int = 8,
):
    """Main evaluation pipeline matching original RewardBench2 implementation.

    Supports both four-way comparison (non-Ties) and absolute rating (Ties) modes.

    Args:
        data_path: Path to input dataset file
        result_path: Path for saving evaluation results
        max_samples: Maximum number of samples to process (-1 for all)
        model: Model identifier string or configuration dictionary
        max_workers: Maximum number of parallel workers for evaluation
    """
    try:
        # Validate input parameters
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

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
                f"Invalid model type: {type(model)}. Expected str or dict."
            )

        # Load evaluation dataset
        dataset = load_module.run()
        samples = dataset.get_data_samples()
        print(f"Loaded {len(samples)} samples for evaluation")

        if not samples:
            print("No samples loaded. Check data file and configuration.")
            return

        # Create evaluator instance
        evaluator = RewardBench2Evaluator(
            reward=RewardBench2Reward(
                name="rewardbench2",
                llm=llm,
            )
        )

        # Execute evaluation pipeline with parallel processing
        results = evaluator.run(samples=samples, max_workers=max_workers)

        # Print detailed evaluation results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nModel: {results.get('model', 'Unknown')}")

        # Print overall accuracy
        overall_acc = results.get("overall_accuracy", {})
        print("\nOverall Performance:")
        print(
            f"  Accuracy: {overall_acc.get('accuracy', 0):.4f} ({overall_acc.get('accuracy', 0)*100:.2f}%)"
        )
        print(
            f"  Correct: {overall_acc.get('correct_count', 0)}/{overall_acc.get('valid_samples', 0)}"
        )
        print(f"  Total samples: {overall_acc.get('total_samples', 0)}")
        print(f"  Non-Ties samples: {overall_acc.get('non_ties_samples', 0)}")
        print(f"  Ties samples: {overall_acc.get('ties_samples', 0)}")

        # Print detailed Ties metrics if available
        if overall_acc.get("ties_valid_count", 0) > 0:
            print("\nTies Evaluation Details:")
            print(
                f"  Basic Accuracy: {overall_acc.get('ties_basic_accuracy', 0):.4f} ({overall_acc.get('ties_basic_accuracy', 0)*100:.2f}%)"
            )
            print(
                f"  Margin Reasonable: {overall_acc.get('ties_margin_reasonable', 0):.4f} ({overall_acc.get('ties_margin_reasonable', 0)*100:.2f}%)"
            )
            print(
                f"  Strict Correct: {overall_acc.get('ties_strict_correct', 0):.4f} ({overall_acc.get('ties_strict_correct', 0)*100:.2f}%)"
            )
            print(f"  Valid Ties samples: {overall_acc.get('ties_valid_count', 0)}")

        # Print subset accuracy
        subset_acc = results.get("subset_accuracy", {})
        if subset_acc:
            print("\nSubset Performance:")
            for subset, metrics in subset_acc.items():
                accuracy = metrics.get("accuracy", 0)
                correct = metrics.get("correct_count", 0)
                valid = metrics.get("valid_samples", 0)
                total = metrics.get("total_samples", 0)
                print(
                    f"  {subset:15s}: {accuracy:.4f} ({accuracy*100:5.2f}%) - {correct:2d}/{valid:2d} correct, {total:2d} total"
                )

                # If this is ties subset, show detailed metrics
                if subset.lower() == "ties" and metrics.get("ties_valid_count", 0) > 0:
                    basic_acc = metrics.get("ties_basic_accuracy", 0)
                    margin_reasonable = metrics.get("ties_margin_reasonable", 0)
                    strict_correct = metrics.get("ties_strict_correct", 0)
                    print(
                        f"    ├─ Basic Accuracy: {basic_acc:.4f} ({basic_acc*100:5.2f}%)"
                    )
                    print(
                        f"    ├─ Margin Reasonable: {margin_reasonable:.4f} ({margin_reasonable*100:5.2f}%)"
                    )
                    print(
                        f"    └─ Strict Correct: {strict_correct:.4f} ({strict_correct*100:5.2f}%)"
                    )

        print("\n" + "=" * 80)

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


if __name__ == "__main__":
    fire.Fire(main)
