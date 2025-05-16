from typing import List

from pydantic import Field

from rm_gallery.pipeline.node.base import Node


class BasePrompt(Node):
    desc: str = Field(default="", description="评测任务描述")

    def run(self):
        ...


class PrincipledPrompt(BasePrompt):
    llm: str = Field(default=...)
    principles: List[str] = Field(default=...)

    def format(self, data) -> str:
        ...

    def run(self, data):
        prompt = self.format(**data)
        response = self.llm.chat(prompt)
        return response
        

class AutoPrincipledPrompt(BasePrompt):
    ...