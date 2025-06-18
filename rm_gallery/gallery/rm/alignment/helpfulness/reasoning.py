from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Reasoning: Involves processing and analyzing text to draw inferences, make predictions, or solve problems, requiring an understanding of underlying concepts and relationships within the text."
PRINCIPLES = [
    "Direct Problem Solving with Explicit Assumptions: Prioritize addressing the core ambiguity or problem head-on by making reasonable, clearly stated assumptions when data is missing, enabling actionable answers.",
    "Logical Consistency and Accuracy: Maintain coherent reasoning within the chosen analytical framework, avoiding contradictions, oversimplifications, or errors in calculations/formulas.",
    "Contextual Relevance and Practical Utility: Align solutions with the user's explicit goals, real-world applicability, and domain-specific constraints (e.g., audience expectations, technical feasibility).",
]


@RewardRegistry.register("reasoning_listwise_reward")
class ReasoningListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
