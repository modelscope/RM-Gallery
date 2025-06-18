from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Open QA: Search for answers across a wide range of text sources. The challenge is to process large amounts of information and understand complex questions."
PRINCIPLES = [
    "Accuracy and Factual Correctness: Ensure all information strictly aligns with verified facts, canonical sources, or scholarly consensus to maintain credibility and avoid misinformation.",
    "Structured Organization: Present information in a logical, coherent framework (chronological, hierarchical, or thematic) to enhance readability and user comprehension.",
    "Relevance to Core Query: Prioritize content directly addressing the user's explicit request while filtering out tangential or redundant details to maintain focus.",
]


@RewardRegistry.register("open_qa_listwise_reward")
class OpenQAListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
