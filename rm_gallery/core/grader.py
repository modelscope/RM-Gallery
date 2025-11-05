import asyncio
import inspect
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.dataset import DataSample, DataSampleMapping
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.template import ChatTemplate


class GraderMode(str, Enum):
    """Grader modes for grader functions.

    Attributes:
        POINTWISE: Pointwise evaluation mode.
        LISTWISE: Listwise evaluation mode.
    """

    POINTWISE = "pointwise"
    LISTWISE = "listwise"


class GraderResult(BaseModel):
    """Base class for grader results.

    This Pydantic model defines the structure for grader results,
    which include a reason and optional metadata.
    """

    reason: str = Field(default=..., description="The reason for the result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="The metadata of the grader result"
    )


class GraderScore(GraderResult):
    """Grader score result.

    Represents a numerical score assigned by a grader along with a reason.
    """

    score: float = Field(default=..., description="score")


class GraderRank(GraderResult):
    """Grader rank result.

    Represents a ranking of items assigned by a grader along with a reason.
    """

    rank: List[int] = Field(default=..., description="rank")


class GraderError(GraderResult):
    """Grader error result.

    Represents an error encountered during evaluation.
    """


class Grader(ABC):
    """Base class for graders.

    This abstract base class defines the interface for all graders.
    Subclasses must implement the evaluate method.

    Attributes:
        name (str): The name of the grader.
        evaluation_mode (GraderMode): The evaluation mode (pointwise or listwise).
    """

    def __init__(
        self, name: str = "", evaluation_mode: GraderMode = GraderMode.POINTWISE
    ):
        """Initialize a Grader.

        Args:
            name: The name of the grader.
            evaluation_mode: The evaluation mode. Defaults to POINTWISE.
        """
        self.name = name
        self.evaluation_mode = evaluation_mode

    def __name__(self):
        """Get the name of the grader.

        Returns:
            str: The name of the grader.
        """
        return self.name

    @abstractmethod
    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate method to be implemented by subclasses.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.
        """
        ...

    async def run(self, *args, **kwargs) -> GraderScore | GraderRank | GraderError:
        """Run the grader.
        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            GraderScore | GraderRank | GraderError: The evaluation result.
        """
        try:
            return await self.run(*args, **kwargs)
        except Exception as e:
            error = f"Error in {self.name}: {e}"
            logger.error(error)
            return GraderError(reason=str(e))

    async def __call__(
        self, data_sample: DataSample, *args, **kwargs
    ) -> List[GraderScore]:
        """Evaluate based on the specified evaluation mode.

        Args:
            data_sample: The data sample to evaluate.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List of grader scores.

        Raises:
            ValueError: If the evaluation mode is invalid.
        """
        if self.evaluation_mode == GraderMode.POINTWISE:
            # Pointwise: Evaluate each sample individually
            coroutines = [
                self.evaluate(**data_sample.data, **sample)
                for sample in data_sample.samples
            ]
            results: List[GraderScore] = await asyncio.gather(*coroutines)  # type: ignore
            return results

        elif self.evaluation_mode == GraderMode.LISTWISE:
            # Listwise: Evaluate all samples together in one call
            params = {key: value for key, value in kwargs.items()}
            if len(data_sample.samples) > 1:
                if data_sample.samples:
                    for key in data_sample.samples[0].keys():
                        params[key] = "\n".join(
                            [
                                f"Sample {i+1}: {sample[key]}"
                                for i, sample in enumerate(data_sample.samples)
                            ]
                        )

            result = await self.evaluate(**params)
            assert isinstance(result, GraderRank)
            result_list = [
                GraderScore(
                    score=score,
                    reason=result.reason,
                )
                for score in result.rank
            ]
            return result_list
        else:
            raise ValueError(f"Invalid evaluation mode: {self.evaluation_mode}")


class LLMGrader(Grader):
    """LLM-based evaluation grader.

    A grader that uses a large language model to perform evaluations.

    Attributes:
        name (str): The name of the grader.
        evaluation_mode (GraderMode): The evaluation mode.
        chat (ChatTemplate): The chat template for the LLM.
        kwargs (dict): The kwargs for the grader.
    """

    def __init__(
        self,
        name: str = "",
        evaluation_mode: GraderMode = GraderMode.POINTWISE,
        chat: ChatTemplate | dict | None = None,
        **kwargs,
    ):
        """Initialize an LLMGrader.

        Args:
            name: The name of the grader.
            evaluation_mode: The evaluation mode.
            chat: The chat template for the LLM.
            kwargs: The kwargs for the grader.
        """
        super().__init__(name, evaluation_mode)
        if isinstance(chat, dict):
            chat = ChatTemplate(**chat)
        self.chat = chat
        self.kwargs = kwargs

    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using LLM.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            Evaluation result (score or rank).

        Raises:
            ValueError: If the evaluation mode is unsupported.
        """
        if self.evaluation_mode == GraderMode.LISTWISE:
            model_output = GraderRank
        else:
            model_output = GraderScore

        # Check if chat is not None before calling it
        if self.chat is None:
            raise ValueError("Chat template is not set")

        response = await self.chat(model_output=model_output, **kwargs, **self.kwargs)
        if self.evaluation_mode == GraderMode.LISTWISE:
            result = GraderRank(
                rank=response.metadata["rank"],  # type: ignore
                reason=response.metadata["reason"],  # type: ignore
            )
        elif self.evaluation_mode == GraderMode.POINTWISE:
            result = GraderScore(
                score=response.metadata["score"],  # type: ignore
                reason=response.metadata["reason"],  # type: ignore
            )
        else:
            raise ValueError(f"Unsupported evaluation mode: {self.evaluation_mode}")
        return result


class FunctionGrader(Grader):
    """Function-based grader.

    A grader that uses a provided function to perform evaluations.

    Attributes:
        func (Callable): The function to use for evaluation.
        name (str): The name of the grader.
        evaluation_mode (GraderMode): The evaluation mode.
    """

    def __init__(
        self,
        func: Callable,
        name: str = "",
        evaluation_mode: GraderMode = GraderMode.POINTWISE,
    ):
        """Initialize a FunctionGrader.

        Args:
            func: The function to use for evaluation.
            name: The name of the grader.
            evaluation_mode: The evaluation mode.
        """
        super().__init__(name, evaluation_mode)
        self.func = func

    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using a function.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            Evaluation result (score or rank).

        Raises:
            TypeError: If result type doesn't match evaluation mode.
        """
        result = await self.func(**kwargs)

        # Check return type based on evaluation mode
        if self.evaluation_mode == GraderMode.POINTWISE:
            if not isinstance(result, GraderScore):
                raise TypeError(
                    f"Expected GraderScore for pointwise mode, got {type(result)}"
                )
        elif self.evaluation_mode == GraderMode.LISTWISE:
            if not isinstance(result, GraderRank):
                raise TypeError(
                    f"Expected GraderRank for listwise mode, got {type(result)}"
                )
        else:
            raise ValueError(f"Unsupported evaluation mode: {self.evaluation_mode}")

        return result


GraderType = Grader | Callable


async def evaluate(
    grader: Callable,
    mapping: DataSampleMapping | Callable | None,
    data_sample: DataSample,
    *args,
    **kwargs,
) -> List[GraderScore]:
    """Evaluate a data sample using a grader.

    Args:
        grader: The grader function to use.
        mapping: Data sample mapping function.
        data_sample: The data sample to evaluate.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        List of grader scores.

    Raises:
        ValueError: If grader function signature is invalid.
    """
    # Check that function has at least one parameter and first parameter is data_sample
    sig = inspect.signature(grader)
    params = list(sig.parameters.keys())

    if not params:
        raise ValueError(f"Function {grader.__name__} must have at least one parameter")

    if "data_sample" not in params:
        raise ValueError(
            f"Function {grader.__name__} must have 'data_sample' as its first parameter"
        )

    return await grader(
        data_sample=mapping(data_sample) if mapping is not None else data_sample,
        *args,
        **kwargs,
    )


class FactualGrader(LLMGrader):
    """Factual evaluation grader.

    A specific implementation of LLMGrader for factual accuracy evaluation.
    """

    def __init__(self):
        """Initialize a FactualGrader with a predefined chat template."""
        chat_template = ChatTemplate(
            template=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are a helpful assistant that evaluates the quality of a "
                        "response. Your job is to evaluate the quality of the response "
                        "and give a score between 0 and 1. The score should be based on "
                        "the quality of the response. The higher the score, the better "
                        "the response. The score should be a number between 0 and 1"
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Please evaluate the quality of the response provided by the "
                        "assistant.\nThe user question is: {query}\nThe assistant "
                        "response is: {answer}\n\nPlease output as the following json "
                        'object:\n{\n    "score": <score>,\n    "reason": <reason>\n}'
                    ),
                ),
            ],
            model={
                "model_name": "qwen-plus",
                "stream": False,
                "client_args": {
                    "timeout": 60,
                },
            },
        )

        super().__init__(
            name="factual_grader",
            evaluation_mode=GraderMode.POINTWISE,
            chat=chat_template,
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
