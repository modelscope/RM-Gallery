from typing import List, Type
from pydantic import Field
from src.model.base import LLMClient
from src.rm.module import LLMMixin, PointRewardModule
from src.rm.schema import LLMModuleOutput
from src.rm.template import BaseTemplate
from src.rm.schema import LLMLLMModuleOutput


class HelpfulnessSchema(LLMLLMModuleOutput):
    helpfulness: str = Field(default=..., description="is answer helpful? Yes or No.")


class HelpfulnessTemplate(BaseTemplate):
    @classmethod
    def format(cls, desc: str, context: str, answer: str, output: Type[LLMLLMModuleOutput]) -> str:
        return f"""# Task Description
        {desc}

        # Context
        {context}

        # Answer
        {answer}

        # Output Format
        {output.format()}
        """


class PrepareData(RuleRewardModule):
    def run(self, sample: DataSample):
        ...


class HelpfulnessReward(PointRewardModule, LLMMixin):
    name: str = Field(default=...)

    # input: List[InputVar] = Field(
    #     default=[
    #         InputVar(name="query", path="history", level=ModuleLevel.LISTWISE),
    #         InputVar(name="context", path="ontext", level=ModuleLevel.LISTWISE),
    #         InputVar(name="answer", path="answer", level=ModuleLevel.POINTWISE)],
    #     description="input vars mapping"
    # )
    output: Type[LLMModuleOutput] | Type[dict] = Field(default=HelpfulnessSchema)

    desc: str | None = Field(default="Your task is to judge whether the answer is helpfull")
    llm: LLMClient = Field(default=..., description="llm client")
    template: Type[BaseTemplate] | str | dict = Field(default=HelpfulnessTemplate)

    def prepare(self, input: InputContext, output: OutputContext) -> dict:
        return super().prepare(input, output)
