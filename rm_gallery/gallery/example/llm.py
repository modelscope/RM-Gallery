# Grader 开发示例
import asyncio
from typing import List

from loguru import logger

from rm_gallery.core.data import DataSample
from rm_gallery.core.grader import GraderMode, LLMGrader, evaluate

DEFAULT_TEMPLATE = {
    "messages": [
        dict(
            role="system",
            content=(
                "You are a helpful assistant that evaluates the quality of a "
                "response. Your job is to evaluate the quality of the response "
                "and give a score between 0 and 1. The score should be based on "
                "the quality of the response. The higher the score, the better "
                "the response. The score should be a number between 0 and 1"
            ),
        ),
        dict(
            role="user",
            content=(
                "Please evaluate the quality of the response provided by the "
                "assistant.\nThe user question is: {query}\nThe assistant "
                "response is: {answer}\n\nPlease output as the following json "
                "object:\n{{\n    score: <score>,\n    reason: <reason>\n}}"
            ),
        ),
    ],
    "required_fields": [
        {
            "name": "query",
            "type": "string",
            "position": "data",
            "description": "The user question in data",
        },
        {
            "name": "answer",
            "type": "string",
            "position": "sample",
            "description": "The assistant response in sample",
        },
    ],
}

DEFAULT_MODEL = {
    "model_name": "qwen-plus",
    "stream": False,
    "client_args": {
        "timeout": 60,
    },
}


class FactualGrader(LLMGrader):
    """Factual evaluation grader.

    A specific implementation of LLMGrader for factual accuracy evaluation.
    """

    def __init__(
        self,
        name="factual_grader",
        evaluation_mode=GraderMode.POINTWISE,
        template: List[dict] | dict = DEFAULT_TEMPLATE,
        model: dict = DEFAULT_MODEL,
    ):
        """Initialize a FactualGrader with a predefined chat template."""
        super().__init__(
            name=name, grader_mode=evaluation_mode, template=template, model=model
        )


def test_factual_grader():
    """Test the factual grader."""
    grader = FactualGrader()

    data_sample = DataSample(
        data={"query": "What is the capital of France?"},
        samples=[{"answer": "Paris"}, {"answer": "London"}],
    )

    result = asyncio.run(
        evaluate(
            grader,
            mapping=None,
            data_sample=data_sample,
        )
    )
    logger.info(result)


if __name__ == "__main__":
    test_factual_grader()
