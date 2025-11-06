from rm_gallery.core.grader import GraderMode
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessGrader

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""
RUBRICS = """"""


class CodeGrader(BaseHelpfulnessGrader):
    """Code: Evaluates the correctness, readability, and efficiency of code-related responses."""

    def __init__(
        self,
        name: str = "Code",
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
