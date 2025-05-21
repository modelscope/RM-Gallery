from typing import List, Type
from pydantic import Field
from src.model.base import LLMClient
from src.rm.base import BaseParser
from src.rm.schema import EvaluationContext
from src.rm.scorer import Rule
from src.rm.template import BaseTemplate, EvaluationTemplate, ParserTemplate


class LLMParser(BaseParser):
    client: LLMClient = Field(default=..., description="llm client")
    desc: str | None = Field(default=None, description="evaluation task description")
    output_schema: Type[EvaluationContext] = Field(default=...)
    template: Type[BaseTemplate] | str = Field(default=ParserTemplate)

    def format(self, **kwargs) -> str:
        return self.template.format(desc=self.desc, output_schema=self.output_schema, **kwargs)

    def _run(self, **kwargs):
        query = self.format(**kwargs)
        response = self.client.simple_chat(query=query)
        output = self.output_schema.parse(response)
        return output


class LLMEvaluation(LLMParser):
    principles: List[str] | str | None = Field(default=None, description="evaluation priciples")
    examples: List[str] | None = Field(default=None, description="evaluation examples")
    template: Type[BaseTemplate] | str = Field(default=EvaluationTemplate)
    rules: List[Rule] | None = Field(default=None)

    def format(self, **kwargs) -> str:
        return self.template.format(desc=self.desc, principles=self.principles, examples=self.examples, output_schema=self.output_schema, rules=self.rules, **kwargs)