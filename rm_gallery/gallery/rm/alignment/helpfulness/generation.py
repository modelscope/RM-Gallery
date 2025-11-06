from rm_gallery.core.grader import GraderMode
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessGrader

RUBRICS = """Adherence to Instructional Specificity: Prioritize addressing all explicit requirements (e.g., format, content scope, tone) with precise alignment to ensure completeness and fidelity to the task's intent.
Depth and Originality in Content: Deliver nuanced, actionable insights or creative elements that exceed generic responses through specific examples, contextual relevance, and imaginative elaboration.
Structural Coherence and Logical Flow: Maintain organized progression (e.g., clear hierarchy, thematic sequencing) to enhance readability while avoiding contradictions or deviations from established frameworks."""

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for your judgment. If you find any of the rubrics helpful for the current problem, feel free to use them as supplements.
"""


class GenerationGrader(BaseHelpfulnessGrader):
    """Generation: Evaluates the quality of open-ended text generation tasks."""

    def __init__(
        self,
        name: str = "Generation",
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
