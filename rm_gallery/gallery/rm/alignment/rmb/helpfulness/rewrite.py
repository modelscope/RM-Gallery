from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Adherence to Explicit Instructions: Prioritize strict compliance with all user-specified modifications while maintaining the original intent and core content.",
    "Preservation of Original Intent: Ensure rephrased content retains the core meaning, purpose, and factual integrity of the original input without unwarranted alterations.",
    "Clarity and Conciseness: Simplify complex structures and eliminate redundancy while preserving technical accuracy and avoiding unnecessary expansion.",
    "Factual and Contextual Accuracy: Ground revisions in empirical evidence, scientific understanding, or genre-specific conventions to maintain credibility and authenticity.",
    "Avoid Unverified Assumptions: Refrain from speculative interpretations or additions not explicitly supported by the input or context.",
]

SCENARIO = """the assitant aims to modifies existing text to alter its style while preserving the original information and intent"""


@RewardRegistry.register("rewrite_listwise_reward")
class RewriteListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
