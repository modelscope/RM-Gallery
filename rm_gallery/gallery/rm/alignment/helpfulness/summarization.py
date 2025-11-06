from rm_gallery.core.grader import GraderMode
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessGrader

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance. You must check each candidate answer in turn to see if it violates any rubric, and provide reasons for any violations you find. These reasons should be used as references for ranking the answers.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.
"""

RUBRICS = """Comprehensive Coverage of Core Content: A superior summary captures all critical elements, themes, and details central to the source material without omitting key information.
Avoidance of Irrelevant or Tangential Information: Focuses exclusively on the primary subject, eliminating extraneous details that distract from the core narrative or argument.
Logical Structure and Coherence: Information is organized in a clear, hierarchical, or chronological sequence to ensure readability and logical progression of ideas.
"""


class SummarizationGrader(BaseHelpfulnessGrader):
    """Summarization: Evaluates the quality of text summarization in terms of coverage, conciseness, and clarity."""

    def __init__(
        self,
        name: str = "Summarization",
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
