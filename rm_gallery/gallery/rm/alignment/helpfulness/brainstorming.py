from rm_gallery.core.grader import GraderMode
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessGrader

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""

RUBRICS = """Creative Relevance and Contextual Alignment: Prioritize completions that balance novel ideas with direct ties to the scenario's core context, ensuring ideas are both imaginative and grounded in the specific problem or theme.
Practical Feasibility and Actionable Detail: Favor completions that offer concrete, implementable solutions or insights, avoiding abstract or overly speculative suggestions that lack real-world applicability.
Structural Coherence and Logical Organization: Prefer completions that present ideas in a clear, logically sequenced framework (e.g., categorized sections, step-by-step processes) to enhance readability and development potential.
"""


class BrainstormingGrader(BaseHelpfulnessGrader):
    """Brainstorming: Evaluates the creativity, diversity, and feasibility of idea generation."""

    def __init__(
        self,
        name: str = "Brainstorming",
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
