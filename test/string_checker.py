import asyncio
from typing import List

from loguru import logger

from rm_gallery.core.dataset import DataSample, DataSampleMapping
from rm_gallery.core.grader import (
    FunctionGrader,
    Grader,
    GraderMode,
    GraderScore,
    evaluate,
)


class StringCheckerV1(Grader):
    """String checker grader."""

    name: str = "string_checker"
    evaluation_mode: GraderMode = GraderMode.POINTWISE

    async def evaluate(self, reference_output, target_output) -> GraderScore:
        """Evaluate by comparing reference and target outputs.

        Args:
            reference_output: Reference output to compare against
            target_output: Target output to evaluate

        Returns:
            Grader score with comparison result
        """
        return GraderScore(
            score=1 if reference_output == target_output else 0,
            reason="String checker",
        )


async def string_checker_v2(data_sample: DataSample) -> List[GraderScore]:
    """Directly evaluate the reward.

    Args:
        data_sample: Data sample containing reference and target outputs

    Returns:
        List of grader scores
    """
    return [
        GraderScore(
            score=1 if data_sample.data["reference"] == sample["target_output"] else 0,
            reason="String checker",
        )
        for sample in data_sample.samples
    ]


async def string_checker_v3(reference_output, target_output) -> GraderScore:
    """Function for Function Grader.

    Args:
        reference_output: Reference output to compare against
        target_output: Target output to evaluate

    Returns:
        Grader score with comparison result
    """
    return GraderScore(
        score=1 if reference_output == target_output else 0,
        reason="String checker",
    )


def test_string_checker_v1():
    """Test StringCheckerV1 implementation."""
    grader = StringCheckerV1()
    data_sample = DataSample(
        data={"reference": "Hello World"},
        samples=[{"target_output": "Hello World"}, {"target_output": "Hello"}],
    )
    result = asyncio.run(
        evaluate(
            grader,
            mapping=DataSampleMapping(
                data_mapping={"reference_output": "reference"},
            ),
            data_sample=data_sample,
        )
    )
    logger.info(result)


def test_string_checker_v2():
    """Test string_checker_v2 implementation."""
    data_sample = DataSample(
        data={"reference": "Hello World"},
        samples=[{"target_output": "Hello World"}, {"target_output": "Hello"}],
    )
    result = asyncio.run(
        evaluate(
            string_checker_v2,
            mapping=None,
            data_sample=data_sample,
        )
    )
    logger.info(result)


def test_string_checker_v3():
    """Test string_checker_v3 implementation."""
    data_sample = DataSample(
        data={"reference": "Hello World"},
        samples=[{"target_output": "Hello World"}, {"target_output": "Hello"}],
    )

    grader = FunctionGrader(name="string checker", func=string_checker_v3)
    result = asyncio.run(
        evaluate(
            grader,
            mapping=DataSampleMapping(
                data_mapping={"reference_output": "reference"},
            ),
            data_sample=data_sample,
        )
    )
    logger.info(result)


if __name__ == "__main__":
    test_string_checker_v2()
