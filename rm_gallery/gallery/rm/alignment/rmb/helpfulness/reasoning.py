from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Reasoning: Involves processing and analyzing text to draw inferences, make predictions, or solve problems, requiring an understanding of underlying concepts and relationships within the text."
PRINCIPLES = [
    "Clarify Ambiguities and Avoid Unwarranted Assumptions: Explicitly resolve inherent ambiguities in the scenario and reject speculative premises not supported by given information to prevent flawed reasoning.",
    "Prioritize Logical and Systematic Rigor: Apply structured methodologies, mathematical precision, and step-by-step validation to ensure coherence and avoid oversimplification or errors.",
    "Align with Contextual and Domain-Specific Relevance: Tailor analysis to the scenario's constraints, audience expectations, and technical requirements while avoiding overgeneralization or misaligned strategies.",
    "Evaluate Evidence and Limitations Holistically: Systematically assess available data, credibility of sources, and limitations while acknowledging gaps or uncertainties in the information.",
    "Maintain Technical Accuracy and Precision: Ensure domain-specific correctness in terminology, calculations, and application of principles to uphold reliability and validity.",
]


@RewardRegistry.register("reasoning_listwise_reward")
class ReasoningListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
