from typing import List

from pydantic import Field

from rm_gallery.core.reward.base import (
    BaseListWisePrincipleReward,
    BasePointWisePrincipleReward,
)
from rm_gallery.core.reward.registry import RewardRegistry

HELPFULNESS_PRINCIPLES = [
    "Adherence to Instructions: Strictly follow explicit criteria and formatting requirements to ensure compliance with all specified constraints.",
    "Clarity and Structure: Organize information logically with distinct sections and avoid ambiguity through precise, actionable language.",
    "Audience-Centric Communication: Tailor tone and content to the target audience, using direct engagement and culturally relevant references for inclusivity.",
    "Technical Precision: Apply accurate terminology, mathematical rigor, and logical consistency in explanations and calculations.",
    "Stylistic Cohesion: Maintain thematic and metaphorical consistency while enhancing readability through varied sentence structures and rhetorical devices.",
    "Process Optimization: Streamline workflows by prioritizing efficiency in technical implementations (e.g., batch processing, probability normalization).",
    "Validation and Verification: Cross-check outputs against constraints, ensuring correctness through step-by-step validation and error prevention.",
    "Practical Relevance: Address real-world scenarios with actionable solutions, balancing theoretical depth with immediate applicability.",
    "Conciseness and Focus: Eliminate redundancy by distilling complex ideas into essential components while maintaining clarity and completeness.",
    "Dynamic Engagement: Enhance impact through strategic use of attention-grabbing elements (e.g., energetic language, emojis) without compromising professionalism.",
]


SCENARIO = """The assistant aims to provide helpful and informative responses to users, responding to their queries with relevant and accurate information."""


@RewardRegistry.register("helpfulness_pointwise")
class HelpfulnessPointWiseReward(BasePointWisePrincipleReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=HELPFULNESS_PRINCIPLES)


@RewardRegistry.register("helpfulness_listwise")
class HelpfulnessListWiseReward(BaseListWisePrincipleReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=HELPFULNESS_PRINCIPLES)
