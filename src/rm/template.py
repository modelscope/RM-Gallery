from typing import Type
from pydantic import BaseModel

from src.rm.schema import LLMLLMModuleOutput


class BaseTemplate(BaseModel):
    @classmethod
    def format(cls, output: Type[LLMLLMModuleOutput], **kwargs) -> str:
        ...
