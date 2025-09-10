from abc import abstractmethod
from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BaseReward


class BaseEvaluator(BaseModel):
    reward: BaseReward = Field(
        default=...,
        description="the reward module",
    )

    @abstractmethod
    def summary(self, results: List[DataSample]) -> dict:
        ...

    def run(self, samples: List[DataSample], **kwargs) -> dict:
        samples = self.reward.evaluate_batch(samples=samples)
        metrics = self.summary(samples)
        logger.info(metrics)
        return metrics
