from typing import List

from pydantic import BaseModel, Field


class Sample(BaseModel):
    """Sample data model."""

    data: dict = Field(default_factory=dict, description="data")


class DataSample(BaseModel):
    """Data sample containing data and samples to evaluate."""

    data: dict = Field(default_factory=dict, description="data")
    samples: List[dict] = Field(default_factory=list, description="samples to evaluate")


print(DataSample.model_json_schema())
