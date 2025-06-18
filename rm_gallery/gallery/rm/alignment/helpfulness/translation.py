from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Accuracy in Conveying Original Meaning: Prioritize faithful representation of the source text's intent, details, and nuances without omissions, additions, or distortions.",
    "Proper Terminology and Consistency: Use contextually appropriate, standardized terms and maintain uniformity for recurring concepts to avoid confusion and preserve clarity.",
    "Natural Fluency and Cultural Appropriateness: Ensure translations sound idiomatic and grammatically correct in the target language while adapting expressions to align with cultural conventions.",
]

SCENARIO = "Translation: Converting text from one language to another."


@RewardRegistry.register("translation_listwise_reward")
class TranslationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
