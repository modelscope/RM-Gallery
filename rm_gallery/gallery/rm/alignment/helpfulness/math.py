from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of principles, listed under the heading #Principles. These principles are ordered from highest to lowest importance. You must check each candidate answer in turn to see if it violates any principle, and provide reasons for any violations you find. These reasons should be used as references for ranking the answers.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.
"""
SCENARIO = "Math: Solves problems at math, on open-ended human prompts ranging from middle school physics and geometry to college-level chemistry, calculus, combinatorics, and more."
PRINCIPLES = [
    "Mathematical Accuracy: Ensuring all calculations, formulas, arithmetic operations, and logical deductions are executed correctly to produce precise and valid results.",
    "Correct Application of Mathematical Principles: Prioritizing the accurate use of relevant theorems, definitions, and problem-specific methodologies (e.g., geometric properties, substitution rules, or combinatorial logic) to maintain structural and conceptual validity.",
    "Verification Against Constraints and Answer Choices: Explicitly cross-checking derived results with given problem conditions, provided answer options, or logical consistency to eliminate errors and ensure alignment with requirements.",
]


@RewardRegistry.register("Math_listwise_reward")
class MathListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
