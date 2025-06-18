from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Adherence to Explicit Instructions and Context: Base modifications strictly on provided instructions and source material, avoiding deviations, embellishments, or speculative additions while maintaining contextual relevance."
]

SCENARIO = """Rewrite: the assitant aims to modifies existing text to alter its style while preserving the original information and intent."""


@RewardRegistry.register("rewrite_listwise_reward")
class RewriteListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
