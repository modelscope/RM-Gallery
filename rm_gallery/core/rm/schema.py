from typing import Generic, List, TypeVar

from pydantic import BaseModel, Field


class Dimension(BaseModel):
    name: str = Field(default=..., description="name")
    weight: float = Field(default=..., description="weight")
    reason: str = Field(default=..., description="reason")


class DimensionScore(Dimension):
    score: float = Field(default=..., description="score")


class DimensionRank(Dimension):
    rank: List[float] = Field(default_factory=list, description="rank")

    def __getitem__(self, index: int) -> DimensionScore:
        return DimensionScore(
            name=self.name,
            weight=self.weight,
            reason=self.reason,
            score=self.rank[index],
        )


T = TypeVar("T", DimensionScore, DimensionRank)


class ModuleResult(BaseModel, Generic[T]):
    module_name: str = Field(default=..., description="module name")
    reward_details: List[T] = Field(default_factory=list, description="rewards")
    extra_data: dict = Field(default_factory=dict, description="extra data")
