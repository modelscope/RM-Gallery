"""
Pointwise Evaluator for VERL Integration

Direct scoring mode: each response gets a score from 1-10
"""

import concurrent.futures
from typing import Dict, List

from rm_gallery.core.model.base import BaseLLM
from rm_gallery.gallery.evaluation.llm_judge.templates.base import (
    PointwisePromptTemplate,
)


class PointwiseEvaluator:
    """
    Pointwise evaluator: directly score each response (1-10 scale)
    """

    def __init__(
        self,
        llm: BaseLLM,
        template: PointwisePromptTemplate = None,
        max_workers: int = 10,
        verbose: bool = False,
        **kwargs,
    ):
        self.llm = llm
        self.template = template or PointwisePromptTemplate()
        self.max_workers = max_workers
        self.verbose = verbose
        self.kwargs = kwargs

    def evaluate(
        self, prompt: str, responses: List[str], reference: str = None, **kwargs
    ) -> Dict:
        """
        Execute pointwise evaluation

        Process:
        1. Score each response independently using LLM
        2. Normalize scores to [-1, 1] range
        """
        n = len(responses)

        if n == 0:
            return {"scores": [], "raw_scores": []}

        # Score all responses concurrently
        raw_scores = self._score_all_responses(prompt, responses, reference)

        # Normalize to [-1, 1]
        normalized_scores = self._normalize_scores(raw_scores)

        return {"scores": normalized_scores, "raw_scores": raw_scores, "n_responses": n}

    def _score_all_responses(
        self, prompt: str, responses: List[str], reference: str
    ) -> List[float]:
        """
        Score all responses concurrently using LLM

        Returns:
            List of scores (1-10 scale)
        """
        scores = [5.0] * len(responses)  # Default medium score

        # Concurrent LLM calls
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_idx = {
                executor.submit(
                    self._score_single_response, prompt, responses[i], reference, i
                ): i
                for i in range(len(responses))
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    score = future.result()
                    scores[idx] = score
                except Exception as e:
                    if self.verbose:
                        print(f"Error scoring response {idx}: {e}")
                    scores[idx] = 5.0  # Default medium score

        return scores

    def _score_single_response(
        self, prompt: str, response: str, reference: str, idx: int
    ) -> float:
        """
        Use LLM to score a single response (1-10 scale)
        """
        # Generate prompt
        scoring_prompt = self.template.generate_prompt(
            user_query=prompt,
            response_a=response,
            response_b=None,
            reference=reference or "",
        )

        # Call LLM
        from rm_gallery.core.model.message import ChatMessage, MessageRole

        messages = [ChatMessage(role=MessageRole.USER, content=scoring_prompt)]

        try:
            response_obj = self.llm.chat(messages=messages)
            llm_output = response_obj.message.content
        except Exception as e:
            if self.verbose:
                print(f"LLM call failed for response {idx}: {e}")
            return 5.0  # Default medium score

        # Parse score
        score = self.template.parse_result(llm_output)

        # Ensure score is float
        if isinstance(score, str):
            try:
                score = float(score)
            except:
                score = 5.0

        return float(score)

    def _normalize_scores(self, raw_scores: List[float]) -> List[float]:
        """
        Normalize scores from 1-10 scale to [-1, 1] range

        Note: raw_scores are already guaranteed to be floats from _score_single_response
        """
        if not raw_scores:
            return []

        min_score = min(raw_scores)
        max_score = max(raw_scores)

        # If all scores are the same
        if max_score - min_score < 1e-6:
            return [0.0] * len(raw_scores)

        # Normalize to [-1, 1]
        normalized = [
            2 * (score - min_score) / (max_score - min_score) - 1
            for score in raw_scores
        ]

        return normalized
