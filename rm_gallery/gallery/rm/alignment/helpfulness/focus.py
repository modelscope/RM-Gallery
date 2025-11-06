from rm_gallery.core.grader import GraderMode
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessGrader

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance. You must check each candidate answer in turn to see if it violates any rubric, and provide reasons for any violations you find. These reasons should be used as references for ranking the answers.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.
"""

RUBRICS = """Direct Relevance to Core Query: Prioritize completions that explicitly address the specific question, task, or scenario posed in the query without introducing tangential concepts, unnecessary details, or unrelated analysis."""


class FocusGrader(BaseHelpfulnessGrader):
    """Focus: Evaluates how well responses stay on topic and directly address the user's query."""

    def __init__(
        self,
        name: str = "Focus",
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
