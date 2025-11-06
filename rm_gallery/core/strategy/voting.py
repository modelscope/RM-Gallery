import asyncio
from typing import Callable, List

from loguru import logger

from rm_gallery.core.data import DataSample
from rm_gallery.core.grader import Grader, GraderScore, evaluate
from rm_gallery.core.strategy.base import GraderStrategy
from rm_gallery.gallery.example.llm import FactualGrader


class VotingStrategy(GraderStrategy):
    """Voting grader strategy that optimizes results by executing the grader
    multiple times and averaging the results.
    """

    def __init__(self, grader: Grader | Callable, num_repeats: int = 5, **kwargs):
        """Initialize VotingStrategy.

        Args:
            grader: The grader to optimize
            num_repeats: Number of repetitions, defaults to 5
            **kwargs: Other parameters
        """
        super().__init__(**kwargs)
        self.num_repeats = num_repeats
        self.grader = grader

    def __name__(self) -> str:
        return f"{self.grader.__name__}_voting_{self.num_repeats}"

    async def __call__(
        self, data_sample: DataSample, *args, **kwargs
    ) -> List[GraderScore]:
        """Optimize reward results by voting (repeating execution and averaging).

        Args:
            data_sample: Data sample containing data and samples
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List of optimized reward results with scores averaged over multiple runs
        """
        # Collect all repeated execution tasks
        tasks = [
            self.grader(data_sample, *args, **kwargs) for _ in range(self.num_repeats)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Calculate average scores
        if not results:
            return []

        # Initialize averaged results list
        averaged_results = []
        num_samples = len(results[0])  # Assume all results have the same length

        for i in range(num_samples):
            # Get scores for the i-th sample from all repetitions
            scores = [result[i].score for result in results]
            reasons = [result[i].reason for result in results]

            # Calculate average score
            avg_score = sum(scores) / len(scores)

            # Create new GraderScore with detailed voting information
            averaged_results.append(
                GraderScore(
                    score=avg_score,
                    reason=f"Voting optimization over {self.num_repeats} runs. "
                    f"Individual scores: {scores}, reasons: {reasons}",
                    metadata={
                        f"attempt_{j+1}": result[i] for j, result in enumerate(results)
                    },
                )
            )

        return averaged_results


if __name__ == "__main__":
    data_sample = DataSample(
        data={"query": "What is the capital of France?"},
        samples=[{"answer": "Paris"}, {"answer": "London"}],
    )

    result = asyncio.run(
        evaluate(
            VotingStrategy(FactualGrader()),
            mapping=None,
            data_sample=data_sample,
        )
    )
    logger.info(result)
