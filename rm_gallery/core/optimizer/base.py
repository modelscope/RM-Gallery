from abc import ABC
from typing import Callable, List

from rm_gallery.core.dataset import DataSample
from rm_gallery.core.grader import Grader, GraderScore


class GraderOptimizer(ABC):
    """Base grader optimizer class for optimizing input reward functions.

    This class serves as an abstract base class that defines the basic interface
    for reward optimizers. Subclasses should implement the specific optimization logic.
    """

    def __init__(self, grader: Grader | Callable, **kwargs):
        self.grader = grader

    def __name__(self) -> str:
        return self.__class__.__name__

    async def __call__(
        self, data_sample: DataSample, *args, **kwargs
    ) -> List[GraderScore]:
        """Core method for optimizing reward functions.

        Args:
            data_sample: Data sample containing data and samples
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List of optimized reward results
        """
        ...
