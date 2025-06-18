from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    # "Role Consistency: Maintain the assigned character's traits, voice, and thematic alignment throughout interactions to ensure authenticity and immersion.",
    "Adherence to Instructional Guidelines: Strictly follow all specified structural, formatting, and procedural requirements in the prompt to meet functional expectations.",
    "Interactive and Immersive Engagement: Prioritize dynamic user involvement, contextual richness, and narrative tension to sustain engagement while respecting role boundaries.",
]
SCENARIO = "Role Playing: Entails adopting specific characters or personas within text-based scenarios, engaging in dialogues or actions that reflect the assigned roles."


@RewardRegistry.register("role_playing_listwise_reward")
class RolePlayingListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
