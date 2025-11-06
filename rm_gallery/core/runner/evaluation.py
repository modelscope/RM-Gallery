import asyncio
from typing import List

from loguru import logger

from rm_gallery.core.data import DataSample, validate_data_samples
from rm_gallery.core.grader import Grader, GraderType, evaluate
from rm_gallery.core.runner.base import BaseRunner
from rm_gallery.gallery.example.llm import FactualGrader


class EvaluationRunner(BaseRunner):
    """Runner for evaluating graders."""

    def __init__(self, graders: List[GraderType], mappings: dict = {}):
        """Initialize the EvaluationRunner.

        Args:
            graders: List of graders to use for the experiment
            mappings: Mappers for the graders
        """
        self.graders = graders or []
        self.mappings = mappings or {}

    async def evaluate(self, data_sample: DataSample) -> List[Grader]:
        """Run experiment for a single sample.

        Args:
            data_sample: The data sample to evaluate

        Returns:
            List of evaluation results
        """
        results = []
        coroutines = []
        for grader in self.graders:
            # Get mapper for grader, if not exist use None
            mapping = (
                self.mappings.get(grader.__name__, None)
                if hasattr(grader, "__name__")
                else None
            )
            coro = evaluate(grader, mapping, data_sample)
            coroutines.append(coro)
        results = await asyncio.gather(*coroutines)
        return results

    async def __call__(self, data_samples: List[DataSample], *args, **kwargs) -> dict:
        """Run experiment.

        Args:
            dataset: The evaluation dataset

        Returns:
            Evaluation result
        """
        results = []
        coroutines = []

        # Create async tasks for each data sample
        for data_sample in data_samples:
            coroutines.append(self.evaluate(data_sample))

        # Execute all tasks in parallel
        results = await asyncio.gather(*coroutines)
        logger.info(f"Results: {results}")

        # TODO: summary of results
        return {
            "results": results,
        }


if __name__ == "__main__":
    data_sample_schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query", "answer"],
            },
            "samples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["query", "answer"],
                },
            },
        },
        "required": ["data", "samples"],
    }
    data_samples = [
        {
            "data": {
                "query": "What is the capital of France?",
            },
            "samples": [{"answer": "Paris"}, {"answer": "Marseille"}],
        },
        {
            "data": {
                "query": "What is the capital of Germany?",
            },
            "samples": [{"answer": "Berlin"}, {"answer": "Munich"}],
        },
    ]
    data_samples = validate_data_samples(data_samples, data_sample_schema)
    runner = EvaluationRunner(graders=[FactualGrader()])
    # Run using async method
    result = asyncio.run(runner(data_samples=data_samples))
