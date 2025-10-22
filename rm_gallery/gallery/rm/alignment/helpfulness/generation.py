from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

SCENARIO = "Generation: Creating new textual content, from articles to stories, with an emphasis on originality and creativity."
RUBRICS = [
    "Adherence to Instructional Specificity: Prioritize addressing all explicit requirements (e.g., format, content scope, tone) with precise alignment to ensure completeness and fidelity to the task's intent.",
    "Depth and Originality in Content: Deliver nuanced, actionable insights or creative elements that exceed generic responses through specific examples, contextual relevance, and imaginative elaboration.",
    "Structural Coherence and Logical Flow: Maintain organized progression (e.g., clear hierarchy, thematic sequencing) to enhance readability while avoiding contradictions or deviations from established frameworks.",
]

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for your judgment. If you find any of the rubrics helpful for the current problem, feel free to use them as supplements.
"""


@RewardRegistry.register("generation_listwise_reward")
class GenerationListWiseReward(BaseHelpfulnessListWiseReward):
    """Generation: Creating new textual content, from articles to stories, with an emphasis on originality and creativity."""

    name: str = Field(default="generation_listwise_reward", description="reward name")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    rubrics: List[str] = Field(default=RUBRICS)
    desc: str = Field(default=DESC)
