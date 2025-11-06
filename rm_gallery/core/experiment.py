import asyncio
from typing import List

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from rm_gallery.core.dataset import DataSample, EvaluationDataset
from rm_gallery.core.grader import Grader, GraderType, evaluate
from rm_gallery.gallery.example.llm import FactualGrader


class EvaluationResult(BaseModel):
    """Analysis result for experiment."""


class EvaluationExperiment(BaseModel):
    """Experiment for evaluating graders."""

    graders: List[GraderType] = Field(
        default_factory=list, description="graders to use for the experiment"
    )
    mappings: dict = Field(default_factory=dict, description="mappers for the graders")

    @model_validator(mode="before")
    def validate_graders(cls, values):
        """Validate graders.

        Args:
            values: Dictionary of values to validate

        Returns:
            Validated values
        """
        graders = values.get("graders")
        for i, grader in enumerate(graders):
            if isinstance(grader, str):
                graders[i] = grader
        return values

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
                if hasattr(grader, "name")
                else None
            )
            coro = evaluate(grader, mapping, data_sample)
            coroutines.append(coro)
        results = await asyncio.gather(*coroutines)
        return results

    async def __call__(self, dataset: EvaluationDataset) -> EvaluationResult:
        """Run experiment.

        Args:
            dataset: The evaluation dataset

        Returns:
            Evaluation result
        """
        results = []
        coroutines = []

        # Create async tasks for each data sample
        for data_sample in dataset.data_samples:
            coroutines.append(self.evaluate(data_sample))

        # Execute all tasks in parallel
        results = await asyncio.gather(*coroutines)
        logger.info(f"Results: {results}")

        # TODO: summary of results

        return EvaluationResult()


if __name__ == "__main__":
    dataset = EvaluationDataset(
        data_sample_schema={
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
        },
        data_samples=[
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
        ],
    )
    experiment = EvaluationExperiment(graders=[FactualGrader()])
    # Run using async method
    result = asyncio.run(experiment(dataset))
