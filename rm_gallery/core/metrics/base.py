"""
Base Metric Classes

Define base abstract classes for evaluation metrics, following RM-Gallery design patterns.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from loguru import logger
from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.metrics.schema import (
    BatchComparisonInput,
    ComparisonInput,
    MetricResult,
)


class BaseMetric(BaseModule, ABC):
    """
    Base class for text metrics, following RM-Gallery design patterns

    All concrete evaluation metrics should inherit from this class and implement the compute method.

    Attributes:
        name: Metric name for identification and registration
        normalize_text: Whether to normalize input text
        case_sensitive: Whether to be case-sensitive (some metrics may require this)
        max_workers: Maximum number of parallel worker threads for batch processing

    Example:
        >>> class MyMetric(BaseMetric):
        ...     name: str = "my_metric"
        ...
        ...     def compute(self, input_data: ComparisonInput) -> MetricResult:
        ...         # Implement specific computation logic
        ...         score = calculate_score(input_data.reference, input_data.candidate)
        ...         return MetricResult(name=self.name, score=score)
    """

    name: str = Field(..., description="Metric name")
    normalize_text: bool = Field(default=True, description="Whether to normalize text")
    case_sensitive: bool = Field(
        default=False, description="Whether to be case-sensitive"
    )
    max_workers: int = Field(default=8, description="Maximum number of worker threads")

    @abstractmethod
    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute metric score for a single sample

        This is the core method that all subclasses must implement.

        Args:
            input_data: Comparison input containing reference and candidate text

        Returns:
            MetricResult: Evaluation result containing score and details

        Raises:
            NotImplementedError: If subclass doesn't implement this method

        Example:
            >>> metric = MyMetric()
            >>> input_data = ComparisonInput(
            ...     reference="hello world",
            ...     candidate="hello world"
            ... )
            >>> result = metric.compute(input_data)
            >>> print(result.score)
            1.0
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute()")

    def compute_batch(
        self, input_data: BatchComparisonInput, max_workers: Optional[int] = None
    ) -> List[MetricResult]:
        """
        Compute metrics in batch (with parallel processing support)

        Decomposes batch input into multiple individual computation tasks and executes them in parallel using a thread pool.

        Args:
            input_data: Batch comparison input data
            max_workers: Maximum number of worker threads, None uses instance configured value

        Returns:
            List[MetricResult]: List of evaluation results for each sample

        Example:
            >>> metric = MyMetric()
            >>> batch_input = BatchComparisonInput(
            ...     references=["ref1", "ref2"],
            ...     candidates=["cand1", "cand2"]
            ... )
            >>> results = metric.compute_batch(batch_input)
            >>> len(results)
            2
        """
        if not input_data.validate_length():
            raise ValueError(
                f"Length mismatch: {len(input_data.references)} references "
                f"vs {len(input_data.candidates)} candidates"
            )

        max_workers = max_workers or self.max_workers
        results = []

        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for ref, cand in zip(input_data.references, input_data.candidates):
                future = executor.submit(
                    self.compute,
                    ComparisonInput(
                        reference=ref,
                        candidate=cand,
                        language=input_data.language,
                        normalize=input_data.normalize,
                    ),
                )
                futures.append(future)

            # Collect results (maintain order)
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error computing metric {self.name}: {e}")
                    # Return default failure result
                    results.append(
                        MetricResult(
                            name=self.name,
                            score=0.0,
                            details={"error": str(e)},
                        )
                    )

        return results

    def _normalize(self, text: str) -> str:
        """
        Text normalization processing

        Can be overridden by subclasses to implement custom normalization logic.

        Args:
            text: Text to be normalized

        Returns:
            str: Normalized text
        """
        if not self.normalize_text:
            return text

        from rm_gallery.core.metrics.utils.normalization import normalize_text

        return normalize_text(text, case_sensitive=self.case_sensitive)

    def _handle_multiple_references(
        self, candidate: str, references: List[str]
    ) -> tuple[float, dict]:
        """
        Handle multiple reference texts

        Default strategy: Compute score between candidate and each reference, return the highest score.
        Subclasses can override this method to implement different aggregation strategies.

        Args:
            candidate: Candidate text
            references: List of reference texts

        Returns:
            tuple[float, dict]: (highest score, details dictionary)
        """
        scores = []
        for ref in references:
            # Create single reference input and compute
            single_input = ComparisonInput(
                reference=ref, candidate=candidate, normalize=self.normalize_text
            )
            result = self.compute(single_input)
            scores.append(result.score)

        max_score = max(scores) if scores else 0.0
        details = {
            "scores_per_reference": scores,
            "max_score": max_score,
            "min_score": min(scores) if scores else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
        }

        return max_score, details

    def __call__(self, input_data: ComparisonInput) -> MetricResult:
        """
        Make metric object callable

        Example:
            >>> metric = MyMetric()
            >>> result = metric(input_data)  # Equivalent to metric.compute(input_data)
        """
        return self.compute(input_data)

    def run(self, **kwargs) -> MetricResult:
        """
        Implement BaseModule's run method

        Used for compatibility with RM-Gallery's BaseModule interface.
        """
        if "input_data" in kwargs:
            return self.compute(kwargs["input_data"])
        elif "reference" in kwargs and "candidate" in kwargs:
            input_data = ComparisonInput(
                reference=kwargs["reference"], candidate=kwargs["candidate"]
            )
            return self.compute(input_data)
        else:
            raise ValueError(
                "Must provide either 'input_data' or 'reference' and 'candidate'"
            )


class BaseStringMetric(BaseMetric):
    """
    Base class for string metrics

    Used for exact matching, substring matching and other string-based operations.
    Usually doesn't require complex NLP processing.
    """

    normalize_text: bool = Field(
        default=False, description="String metrics default to no normalization"
    )


class BaseNLPMetric(BaseMetric):
    """
    Base class for NLP metrics

    Used for BLEU, ROUGE, METEOR and other metrics that require tokenization and linguistic processing.
    """

    normalize_text: bool = Field(
        default=True, description="NLP metrics default to normalization"
    )
    use_stemming: bool = Field(default=False, description="Whether to use stemming")
    language: str = Field(default="en", description="Language")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text (can be overridden by subclasses)

        Args:
            text: Text to tokenize

        Returns:
            List[str]: Tokenization result
        """
        from rm_gallery.core.metrics.utils.tokenization import simple_tokenize

        return simple_tokenize(text)
