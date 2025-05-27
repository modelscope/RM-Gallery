import re
from typing import Self, Type
from pydantic import BaseModel, Field


class BaseTemplate(BaseModel):
    @classmethod
    def parse(cls, text: str) -> Self:
        pattern = r'<([^>]+)>(.*)</\1>'
        matches = re.findall(pattern, text, re.DOTALL)
        contents = {match[0]: match[1] for match in matches}
        return cls(**contents)

    @classmethod
    def schema_format(cls) -> str:
        schema_str = ""
        for key, property in cls.model_json_schema(by_alias=True)["properties"].items():
            schema_str += f"<{key}>{property["description"]}</{key}>"
        return schema_str
    
    @classmethod
    def format(cls, **kwargs) -> str:
        ...


class ReasoingTemplate(BaseTemplate):
    reason: str = Field(default=..., description="analysis process", alias="think")


