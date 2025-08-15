"""
RewardBench2 Reward with Principle Support
"""
from typing import Dict, List, Type
from loguru import logger
from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.schema import RewardDimensionWithRank, RewardResult
from rm_gallery.gallery.evaluation.rewardbench2 import RewardBench2Reward
from .rewardbench2_principle import RewardBench2PrincipleTemplate


class RewardBench2PrincipleReward(RewardBench2Reward):
    """RewardBench2 reward model with principle support.

    Extends the original RewardBench2Reward to support custom evaluation principles
    while maintaining compatibility with the original evaluation protocol.
    """

    template: Type[RewardBench2PrincipleTemplate] = Field(
        default=RewardBench2PrincipleTemplate,
        description="Template class for prompt generation and response parsing"
    )

    # Principles for evaluation
    principles: str = Field(
        default="",
        description="Custom evaluation principles to guide the assessment"
    )

    def _evaluate_four_way(self, sample: DataSample, **kwargs) -> RewardResult:
        """Evaluate using four-way comparison mode with principles."""
        import random  # Import here to avoid issues
        
        query = sample.input[-1].content
        answers = self._ensure_four_answers(sample)

        # Find the index of the chosen (correct) answer
        chosen_index = None
        for i, output in enumerate(sample.output[:4]):  # Only check first 4 outputs
            if (hasattr(output.answer, 'label') and
                isinstance(output.answer.label, dict) and
                output.answer.label.get("preference") == "chosen"):
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

        # Format prompt with principles using the new template
        prompt = self.template.format(
            query=query,
            answers=shuffled_answers,
            principles=self.principles,
            enable_thinking=kwargs.get('enable_thinking', False),
            is_ties=False
        )

        # Get LLM judgment
        response_text = self.llm.simple_chat(query=prompt)

        # Parse response
        response = self.template.parse_four_way(response_text)

        # Convert back to original indices
        predicted_index = response.best_index
        letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        correct_letter = letter_map[correct_position_after_shuffle]

        # Check if prediction is correct
        is_correct = (response.best_answer == correct_letter)

        # Create result scores: chosen answer gets 1.0 if predicted correctly
        scores = [0.0] * len(sample.output)
        if is_correct:
            scores[chosen_index] = 1.0  # Chosen answer gets score 1

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name,
                    reason=response.reasoning,
                    rank=scores
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
                "principles": self.principles
            }
        )

    def _evaluate_ties(self, sample: DataSample, **kwargs) -> RewardResult:
        """Evaluate using Ties absolute rating mode with principles."""
        query = sample.input[-1].content
        answers = [output.answer.content for output in sample.output]

        # Identify correct and incorrect answers based on preference labels
        correct_indices = []
        incorrect_indices = []
        for i, output in enumerate(sample.output):
            if (hasattr(output.answer, 'label') and
                isinstance(output.answer.label, dict) and
                output.answer.label.get("preference") == "chosen"):
                correct_indices.append(i)
            else:
                incorrect_indices.append(i)

        # Rate each answer individually
        ratings = []
        rating_details = []

        for i, answer in enumerate(answers):
            prompt = self.template.format(
                query=query,
                answers=[answer],
                principles=self.principles,
                enable_thinking=kwargs.get('enable_thinking', False),
                is_ties=True
            )

            # Get LLM rating
            response_text = self.llm.simple_chat(query=prompt)

            # Parse rating
            response = self.template.parse_ties_rating(response_text)
            ratings.append(response.rating)

            rating_details.append({
                "answer_index": i,
                "rating": response.rating,
                "reasoning": response.reasoning,
                "prompt": prompt,
                "response": response_text,
                "is_correct": i in correct_indices
            })

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
                    rank=scores
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
                "principles": self.principles
            }
        )