from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Label Consistency Enforcement: Maintain uniform application of predefined categories across all text instances to uphold reliability and minimize classification bias.",
    "Ambiguity Resolution Protocol: Develop strategies to effectively classify borderline cases, ensuring accurate labeling even when texts exhibit mixed or unclear categorical indicators.",
]


SCENARIO = "Classification: Entails assigning predefined categories or labels to text based on its content."


@RewardRegistry.register("classification_listwise_reward")
class ClassificationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
