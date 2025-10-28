"""
Listwise Evaluator for VERL Integration

Ranking mode: rank all responses in a single LLM call
"""

from typing import Dict, List

from rm_gallery.core.model.base import BaseLLM
from rm_gallery.gallery.evaluation.llm_judge.templates.base import (
    ListwisePromptTemplate,
)


class ListwiseEvaluator:
    """
    Listwise evaluator: rank all responses in a single call
    """

    def __init__(
        self,
        llm: BaseLLM,
        template: ListwisePromptTemplate = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.llm = llm
        self.template = template or ListwisePromptTemplate()
        self.verbose = verbose
        self.kwargs = kwargs

    def evaluate(
        self, prompt: str, responses: List[str], reference: str = None, **kwargs
    ) -> Dict:
        """
        Execute listwise evaluation

        Process:
        1. Ask LLM to rank all responses in one call
        2. Convert ranking to scores in [-1, 1] range
        """
        n = len(responses)

        if n == 0:
            return {"scores": [], "ranking": []}

        if n == 1:
            return {"scores": [0.0], "ranking": [0]}

        # Get ranking from LLM
        ranking = self._rank_responses(prompt, responses, reference, n)

        # Convert ranking to scores
        scores = self._ranking_to_scores(ranking, n)

        return {"scores": scores, "ranking": ranking, "n_responses": n}

    def _rank_responses(
        self, prompt: str, responses: List[str], reference: str, n: int
    ) -> List[int]:
        """
        Use LLM to rank all responses

        Returns:
            List of response indices in ranked order (0-indexed)
            e.g., [2, 0, 1] means response 2 is best, then 0, then 1
        """
        # Generate prompt
        ranking_prompt = self.template.generate_prompt(
            user_query=prompt, responses=responses, reference=reference or ""
        )

        # Call LLM
        from rm_gallery.core.model.message import ChatMessage, MessageRole

        messages = [ChatMessage(role=MessageRole.USER, content=ranking_prompt)]

        try:
            response = self.llm.chat(messages=messages)
            llm_output = response.message.content
        except Exception as e:
            if self.verbose:
                print(f"LLM call failed: {e}")
            # Default: natural order
            return list(range(n))

        # Parse ranking
        ranking = self.template.parse_result(llm_output, n)

        return ranking

    def _ranking_to_scores(self, ranking: List[int], n: int) -> List[float]:
        """
        Convert ranking to scores in [-1, 1] range

        Args:
            ranking: List of indices in ranked order (best first)
            n: Total number of responses

        Returns:
            List of scores for each response
        """
        scores = [0.0] * n

        # Assign scores based on rank position
        # Best gets highest score, worst gets lowest
        for rank_position, response_idx in enumerate(ranking):
            if n == 1:
                score = 0.0
            else:
                # Linear scale: best = 1.0, worst = -1.0
                score = 1.0 - (2.0 * rank_position / (n - 1))

            scores[response_idx] = score

        return scores
