from rm_gallery.core.grader import GraderMode
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessGrader

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance. You must check each candidate answer in turn to see if it violates any rubric, and provide reasons for any violations you find. These reasons should be used as references for ranking the answers.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.
"""
RUBRICS = """Strict Adherence to Explicit Formatting and Structural Requirements: Prioritize exact compliance with all specified formatting, structural, and technical constraints (e.g., punctuation, indentation, bullet points, word counts) as the primary criterion for evaluating completions.
Clarity, Logical Progression, and Thematic Consistency: Ensure content is coherent, logically structured, and maintains alignment with the scenario's core premise, fulfilling implicit demands for depth, relevance, and narrative or analytical consistency.
"""


class PreciseIFGrader(BaseHelpfulnessGrader):
    """Precise IF: Evaluates how accurately responses follow complex, multi-part instructions."""

    def __init__(
        self,
        name: str = "PreciseIF",
        grader_mode: GraderMode = GraderMode.LISTWISE,
        task_description=DESC,
        rubrics=RUBRICS,
        **kwargs,
    ):
        super().__init__(
            name=name,
            grader_mode=grader_mode,
            task_description=task_description,
            rubrics=rubrics,
            **kwargs,
        )
