from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Generation: Creating new textual content, from articles to stories, with an emphasis on originality and creativity."
PRINCIPLES = [
    "Adherence to Instructional Intent: Prioritize fulfilling explicit requirements and implicit goals stated in the prompt above all other considerations.",
    "Comprehensive Coverage of Core Concepts: Ensure all fundamental elements, themes, and specified details are thoroughly addressed to meet the scope of the task.",
    "Structural Coherence and Logical Progression: Organize content with clear, purposeful sequencing that enhances clarity, engagement, and narrative or analytical flow.",
]


@RewardRegistry.register("generation_listwise_reward")
class GenerationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
