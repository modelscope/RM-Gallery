from typing import List

from rm_gallery.core.data import DataSample, DataSampleMapping
from rm_gallery.core.grader import LLMGrader, evaluate
from rm_gallery.core.runner.base import BaseRunner


class AutoRubrics(BaseRunner):
    def __init__(self, grader: LLMGrader, mapping: DataSampleMapping | None = None):
        self.grader = grader
        self.mapping = mapping

    def evaluate(self, data_samples: List[DataSample], rubrics: str):
        return evaluate(
            self.grader, mapping=self.mapping, data_sample=data_samples, rubrics=rubrics
        )
