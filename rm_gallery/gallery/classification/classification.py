from typing import List

from pydantic import Field

from rm_gallery.core.reward.base import (
    BaseListWisePrincipleReward,
    BasePointWisePrincipleReward,
)
from rm_gallery.core.reward.registry import RewardRegistry

CLASSIFICATION_PRINCIPLES = [
    "Content Categorization: Correctly identifies and categorizes the text ’ s topic, style, or genre, such as accurately classifying an article as technology, art, or business.",
    "Quality and Compliance Assessment: Evaluates whether the text adheres to established standards and regulations, identifying specific areas of non-compliance.",
]


@RewardRegistry.register("classification_pointwise")
class ClassificationPointWiseReward(BasePointWisePrincipleReward):
    principles: List[str] = Field(default=CLASSIFICATION_PRINCIPLES)


@RewardRegistry.register("classification_listwise")
class ClassificationListWiseReward(BaseListWisePrincipleReward):
    principles: List[str] = Field(default=CLASSIFICATION_PRINCIPLES)
