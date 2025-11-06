from rm_gallery.core.grader import GraderMode
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessGrader

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for your judgment. If you find any of the rubrics helpful for the current problem, feel free to use them as supplements.If all answers meet all rubrics, you can judge and choose one answer by yourself.
"""

RUBRICS = """Accuracy in Translation: Faithfully convey the original text's meaning, intent, and nuances without distortion, omission, or addition.
Contextual Appropriateness: Preserve the original context, tone, and purpose while adapting to target language conventions and specified formatting requirements.
"""


class TranslationGrader(BaseHelpfulnessGrader):
    """Translation: Evaluates the accuracy, fluency, and cultural appropriateness of language translations."""

    def __init__(
        self,
        name: str = "Translation",
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
