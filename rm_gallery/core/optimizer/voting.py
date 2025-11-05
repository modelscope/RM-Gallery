import asyncio
from typing import Callable, List

from loguru import logger

from rm_gallery.core.dataset import DataSample
from rm_gallery.core.grader import FactualGrader, Grader, GraderScore, evaluate
from rm_gallery.core.optimizer.base import GraderOptimizer


class VotingOptimizer(GraderOptimizer):
    """Voting grader optimizer that optimizes results by executing the grader
    multiple times and averaging the results.
    """

    def __init__(self, grader: Grader | Callable, num_repeats: int = 5, **kwargs):
        """Initialize VotingOptimizer.

        Args:
            grader: The grader to optimize
            num_repeats: Number of repetitions, defaults to 5
            **kwargs: Other parameters
        """
        super().__init__(grader, **kwargs)
        self.num_repeats = num_repeats

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

            # Combine reasons (simply take the first reason, other strategies are possible)
            combined_reason = reasons[0] + f" (voted over {self.num_repeats} runs)"

            # Create new GraderScore object
            averaged_score = GraderScore(
                score=avg_score,
                reason=combined_reason,
                metadata={
                    f"attempt_{j+1}": result[i] for j, result in enumerate(results)
                },
            )
            averaged_results.append(averaged_score)

        return averaged_results


if __name__ == "__main__":
    data_sample = DataSample(
        data={"query": "What is the capital of France?"},
        samples=[{"answer": "Paris"}, {"answer": "London"}],
    )

    result = asyncio.run(
        evaluate(
            VotingOptimizer(FactualGrader()),
            mapping=None,
            data_sample=data_sample,
        )
    )
    logger.info(result)