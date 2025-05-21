import re
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Self, Set, Any, ClassVar


# class Dimension(str, Enum):
#     HONESTY = "honesty"
#     SAFETY = "safety"
#     HELPFULNESS = "helpfulness"
#     COMMON = "common"

#     @classmethod
#     def get_dimension(cls, name: str) -> str:
#         """Get a dimension by name, return the name if it's a custom dimension"""
#         try:
#             return cls(name).value
#         except ValueError:
#             # If it's not a predefined dimension, return the name as is
#             return name

#     @classmethod
#     def is_valid_dimension(cls, name: str) -> bool:
#         """Check if a dimension name is valid (either predefined or custom)"""
#         return isinstance(name, str) and len(name) > 0


class Reward(BaseModel):
    total_score: float = Field(..., description="total score")
    rewards_detail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of dimension-specific rewards including reason"
    )
    
    def get_reward_info(self, dimension: str) -> Dict[str, Any]:
        """Get reward value and reason for a specific dimension"""
        for reward in self.rewards_detail:
            if reward["dimension"] == dimension:
                return {
                    "value": reward.get("value", 0.0),
                    "reason": reward.get("reason"),
                }
        return {"value": 0.0, "reason": None}
    
    def set_reward(self, dimension: str, value: Any, reason: str = None) -> None:
        """Set reward for a specific dimension with optional reason"""
        # Validate dimension
        # if not Dimension.is_valid_dimension(dimension):
        #     raise ValueError(f"Invalid dimension name: {dimension}")
            
        # # Get normalized dimension name
        # dimension = Dimension.get_dimension(dimension)
        
        # Update or add reward
        reward_dict = {"dimension": dimension, "value": value}
        if reason:
            reward_dict["reason"] = reason
            
        # Update existing reward or add new one
        for reward in self.rewards_detail:
            if reward["dimension"] == dimension:
                reward.update(reward_dict)
                break
        else:
            self.rewards_detail.append(reward_dict)
        
        # Update total score if this is the first reward
        if len(self.rewards_detail) == 1:
            self.total_score = value


class DataInfo(BaseModel):
    domain: str = Field(..., description="domain")
    source: Optional[str] = Field(
        default=None, serialization_alias="source"
    )


class EvaluationContext(BaseModel):
    @classmethod
    def parse(cls, text: str) -> Self:
        pattern = r'<([^>]+)>(.*?)</\1>'
        matches = re.findall(pattern, text)
        contents = {match[0]: match[1] for match in matches}
        return cls(**contents)

    @classmethod
    def format(cls) -> str:
        schema_str = ""
        for key, property in cls.model_json_schema(by_alias=True)["properties"].items():
            schema_str += f"<{key}>{property["description"]}</{key}>"
        return schema_str


class ContentList(BaseModel):
    """Content with its corresponding reward"""
    role: str = Field(..., description="role")
    content: Optional[str] = Field(default=None, description="Output content")
    content_label: Optional[str] = Field(default=None, description="content label")
    rewards: Optional[Reward] = Field(default=None, description="Reward for this output")
    # TODO: add extra schema
    extra_metadata: Optional[Dict] = Field(default=None, serialization_alias="extraMetadata")
    evaluation_contexts: Dict[str, EvaluationContext] = Field(default={})


class ContextList(BaseModel):
    """Context list"""
    context_type: str = Field(..., description="context type")
    context: Optional[str] = Field(default=None, description="context")
    extra_metadata: Optional[Dict] = Field(default=None, serialization_alias="extraMetadata")


class EvaluationSample(BaseModel):
    input: List[ContentList] = Field(
        default_factory=list,
        serialization_alias="input"
    )
    outputs: List[ContentList] = Field(
        default_factory=list,
        serialization_alias="outputs"
    )
    contexts: List[ContextList] = Field(
        default_factory=list,
        serialization_alias="contexts"
    )
    data_info: Optional[DataInfo] = Field(
        default=None, serialization_alias="dataInfo"
    )
    extra_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="extraMetadata"
    )
    evaluation_contexts: Dict[str, EvaluationContext] = Field(default={})
    # TODO: support custom schema