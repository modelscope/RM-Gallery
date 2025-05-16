from typing import List
from pydantic import BaseModel, Field
from rm_gallery.pipeline.node import PaletteBlock


class ConditionScore(BaseModel):
    condition: str = Field(default=...)
    score: float | int | str | PaletteBlock = Field(default=..., description="评分器")

    def get_score(self, data) -> float | int | str:
        if isinstance(self.score, PaletteBlock):
            self.score.run(data)


class BaseScorer(PaletteBlock):
    ...


class ConditionScorer(BaseScorer):
    conditions: List[ConditionScore] = Field(default=[])

    def run(self, condition: str, **kwargs):
        for condition_score in self.conditions:
            if condition == condition_score.condition:
                condition_score.get_score(**kwargs)
        

class ParallelScorer(BaseScorer):
    scorers: List[ConditionScore]