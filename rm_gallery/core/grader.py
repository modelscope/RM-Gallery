import asyncio
import inspect
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.data import DataSample, DataSampleMapping
from rm_gallery.core.model.template import (
    ChatTemplate,
    LanguageEnum,
    RequiredField,
    Template,
)


class GraderMode(str, Enum):
    """Grader modes for grader functions.

    Attributes:
        POINTWISE: Pointwise grader mode.
        LISTWISE: Listwise grader mode.
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
        grader_mode (GraderMode): The grader mode (pointwise or listwise).
        description: The description of the grader.
    """

    def __init__(
        self,
        name: str = "",
        grader_mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
    ):
        """Initialize a Grader.

        Args:
            name: The name of the grader.
            grader_mode: The grader mode. Defaults to POINTWISE.
        """
        self.name = name
        self.grader_mode = grader_mode
        self.description = description

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

    async def _safe_evaluate(self, **kwargs) -> GraderScore | GraderRank | GraderError:
        """Safely evaluate, handling exceptions gracefully.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The grader result or error result.
        """
        try:
            result = await self.evaluate(**kwargs)
        except Exception as e:
            error_msg = f"Error in {self.name} during  evaluation: {str(e)}"
            logger.error(error_msg)
            result = GraderScore(
                reason=error_msg,
            )
        return result

    async def __call__(
        self, data_sample: DataSample, *args, **kwargs
    ) -> List[GraderScore]:
        """Evaluate based on the specified grader mode.

        Args:
            data_sample: The data sample to evaluate.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List of grader scores.

        Raises:
            ValueError: If the grader mode is invalid.
        """
        if self.grader_mode == GraderMode.POINTWISE:
            # Pointwise: Evaluate each sample individually
            coroutines = [
                self._safe_evaluate(**data_sample.data, **sample)
                for sample in data_sample.samples
            ]
            results: List[GraderScore] = await asyncio.gather(*coroutines)  # type: ignore
            _results = []
            for result in results:
                if isinstance(result, GraderScore):
                    _results.append(result)
                elif isinstance(result, GraderError):
                    _results.append(GraderScore(score=0.0, reason=result.reason))
                else:
                    raise ValueError(f"Invalid result type: {type(result)}")
            return results

        elif self.grader_mode == GraderMode.LISTWISE:
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

            result = await self._safe_evaluate(**params)
            if isinstance(result, GraderRank):
                results = [
                    GraderScore(
                        score=score,
                        reason=result.reason,
                    )
                    for score in result.rank
                ]
            elif isinstance(result, GraderError):
                results = [GraderScore(score=0.0, reason=result.reason)]
            else:
                raise ValueError(f"Invalid result type: {type(result)}")

            return results
        else:
            raise ValueError(f"Invalid grader mode: {self.grader_mode}")


class LLMGrader(Grader):
    """LLM-based evaluation grader.

    A grader that uses a large language model to perform evaluations.

    Attributes:
        name (str): The name of the grader.
        grader_mode (GraderMode): The grader mode.
        chat (ChatTemplate): The chat template for the LLM.
        rubrics (str): The rubrics for the evaluation.
        kwargs (dict): The kwargs for the grader.
    """

    def __init__(
        self,
        name: str = "",
        grader_mode: GraderMode = GraderMode.POINTWISE,
        template: Template | None = None,
        model: Dict[str, Any] | None = None,
        rubrics: str = "",
        **kwargs,
    ):
        """Initialize an LLMGrader.

        Args:
            name: The name of the grader.
            grader_mode: The grader mode.
            template: The chat template for the LLM.
            model: The model parameters for the LLM.
            rubrics: The rubrics for the evaluation.
            kwargs: The kwargs for the grader.
        """
        super().__init__(name, grader_mode)
        self.template = template
        self.model = model
        if template is not None and model is not None:
            self.chat = ChatTemplate(template=template, model=model)
        else:
            self.chat = None
        self.kwargs = kwargs
        self.rubrics = rubrics

    @property
    def required_fields(
        self,
    ) -> List[RequiredField] | Dict[LanguageEnum, List[RequiredField]]:
        """Get the required fields for the grader.

        Returns:
            List of required fields.
        """
        if self.chat is None:
            return []
        return self.chat.required_fields

    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using LLM.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            Evaluation result (score or rank).

        Raises:
            ValueError: If the grader mode is unsupported.
        """
        if self.grader_mode == GraderMode.LISTWISE:
            chat_output = GraderRank
        else:
            chat_output = GraderScore

        # Check if chat is not None before calling it
        if self.chat is None:
            raise ValueError("Chat template is not set")
        params = {"rubrics": self.rubrics, **self.kwargs}
        params.update(kwargs)

        response = await self.chat(chat_output=chat_output, **params)
        if self.grader_mode == GraderMode.LISTWISE:
            result = GraderRank(
                rank=response.metadata["rank"],  # type: ignore
                reason=response.metadata["reason"],  # type: ignore
            )
        elif self.grader_mode == GraderMode.POINTWISE:
            result = GraderScore(
                score=response.metadata["score"],  # type: ignore
                reason=response.metadata["reason"],  # type: ignore
            )
        else:
            raise ValueError(f"Unsupported grader mode: {self.grader_mode}")
        return result


class FunctionGrader(Grader):
    """Function-based grader.

    A grader that uses a provided function to perform evaluations.

    Attributes:
        func (Callable): The function to use for evaluation.
        name (str): The name of the grader.
        grader_mode (GraderMode): The grader mode.
    """

    def __init__(
        self,
        func: Callable,
        name: str = "",
        grader_mode: GraderMode = GraderMode.POINTWISE,
    ):
        """Initialize a FunctionGrader.

        Args:
            func: The function to use for evaluation.
            name: The name of the grader.
            grader_mode: The grader mode.
        """
        super().__init__(name, grader_mode)
        self.func = func

    async def evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using a function.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            Evaluation result (score or rank).

        Raises:
            TypeError: If result type doesn't match grader mode.
        """
        result = await self.func(**kwargs)

        # Check return type based on grader mode
        if self.grader_mode == GraderMode.POINTWISE:
            if not isinstance(result, GraderScore):
                raise TypeError(
                    f"Expected GraderScore for pointwise mode, got {type(result)}"
                )
        elif self.grader_mode == GraderMode.LISTWISE:
            if not isinstance(result, GraderRank):
                raise TypeError(
                    f"Expected GraderRank for listwise mode, got {type(result)}"
                )
        else:
            raise ValueError(f"Unsupported grader mode: {self.grader_mode}")

        return result

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a function as a grader.

        Args:
            name: The name of the grader.
            func: The function to register.

        Returns:
            The Callable grader.
        """

        def decorator(func: Callable) -> "FunctionGrader":
            return FunctionGrader(func, name)

        return decorator


GraderType = Grader | Callable


async def evaluate(
    grader: Callable,
    mapping: DataSampleMapping | Callable | None,
    data_sample: DataSample | List[DataSample],
    *args,
    **kwargs,
) -> List[GraderScore] | List[List[GraderScore]]:
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

    if isinstance(data_sample, list):
        corutines = [
            evaluate(grader, mapping, sample, *args, **kwargs) for sample in data_sample
        ]
        return await asyncio.gather(*corutines)

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
