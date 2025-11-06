from rm_gallery.core.grader import GraderMode
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessGrader

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for your judgment, though not necessarily required. First, think independently. Use these rubrics only when unsure about certain answers, selecting specific ones based on the questions and answers.
"""
RUBRICS = "Character and Contextual Fidelity: Prioritize maintaining the assigned character's persona, motivations, and world-building consistency while strictly adhering to the scenario's established rules, terminology, and thematic boundaries to ensure immersive authenticity."


class RolePlayingGrader(BaseHelpfulnessGrader):
    """Role Playing: Evaluates how well responses embody specific characters or personas in role-playing scenarios."""

    def __init__(
        self,
        name: str = "RolePlaying",
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
