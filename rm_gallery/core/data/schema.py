from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.rm.schema import DimensionScore


class Reward(BaseModel):
    """Reward for the data sample"""

    details: List[DimensionScore] = Field(default_factory=list, description="details")

    @property
    def total_score(self) -> float:
        """Get the total score of the reward"""
        return sum(
            dimension.score * dimension.weight for dimension in self.details
        ) / sum(dimension.weight for dimension in self.details)


class Step(ChatMessage):
    """Step in the process"""

    label: Optional[Dict[str, Any]] = Field(default=None, description="label")
    reward: Reward = Field(default=Reward(), description="reward")


class DataOutput(BaseModel):
    """Data output"""

    answer: Step = Field(default=...)
    steps: Optional[List[Step]] = Field(default=None, description="steps")


class DataSample(BaseModel):
    """Data sample"""

    unique_id: str = Field(..., description="Unique identifier for the data")
    input: List[ChatMessage] = Field(default_factory=list, description="input")
    output: List[DataOutput] = Field(default_factory=list, description="output")
    domain: Optional[str] = Field(default=None, description="domain")
    source: Optional[str] = Field(default=None, description="source")
    created_at: datetime = Field(default_factory=datetime.now, description="createdAt")
    metadata: Optional[Dict] = Field(default=None, description="metadata")

    # update data
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid field: {key}")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class BaseDataSet(BaseModel):
    """Base dataset class for managing collections of data"""

    datas: List[DataSample] = Field(
        default_factory=list, description="List of data items"
    )
    name: str = Field(..., description="dataset name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="metadata")

    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self.datas)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[DataSample, List[DataSample]]:
        """Support index-based access to data"""
        return self.datas[index]

    def get_data_samples(self) -> List[DataSample]:
        """Get all evaluation samples from the dataset"""
        return [data for data in self.datas]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary format"""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "datas": [data.dict() for data in self.datas],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDataSet":
        """Create dataset from dictionary format"""
        return cls(
            name=data["name"],
            metadata=data.get("metadata", {}),
            datas=[DataSample(**item) for item in data["datas"]],
        )

    class Config:
        arbitrary_types_allowed = True
