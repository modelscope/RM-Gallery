from typing import List
from pydantic import BaseModel, Field
from src.data.data_schema import EvaluationContext


class Rule(BaseModel):
    desc: str = Field(default=..., description="rule description")
    score: float | int | str = Field(default=..., description="scorer")


class LLMEvaluationContext(EvaluationContext):
    reason: str = Field(default=..., description="anlysis/reasons", alias="think")


class Claims(LLMEvaluationContext):
    claims: str = Field(default=..., description="a list of claims")


class ViolatedPrinciples(LLMEvaluationContext):
    priciples: List[int] = Field(default=..., description="violated principles")


class Score(LLMEvaluationContext):
    score: int | float = Field(default=..., description="score by rule")
