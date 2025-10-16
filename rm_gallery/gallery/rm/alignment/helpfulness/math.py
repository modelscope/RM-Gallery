from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance. These rubrics can serve as supplementary knowledge for your judgment. If you find any of the rubrics helpful for the current problem, feel free to use them as supplements. If all answers meet all rubrics, you can judge and choose one answer by yourself.
"""
SCENARIO = ""
RUBRICS = [
    "Mathematical Accuracy: Ensure all calculations, formula applications, and logical steps are error-free, as even minor inaccuracies (e.g., arithmetic mistakes, misapplied rules) invalidate results despite otherwise correct methodologies."
]


@RewardRegistry.register("math_listwise_reward")
class MathListWiseReward(BaseHelpfulnessListWiseReward):
    """Math: Solves problems at math, on open-ended human prompts ranging from middle school physics and geometry to college-level chemistry, calculus, combinatorics, and more."""

    name: str = Field(default="math_listwise_reward")
    desc: str = Field(default=DESC)
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    rubrics: List[str] = Field(default=RUBRICS)
