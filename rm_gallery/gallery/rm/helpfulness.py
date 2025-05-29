from typing import Type

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.rm.module import LLMModule, PointModule
from rm_gallery.core.rm.schema import DimensionScore, ModuleResult
from rm_gallery.core.rm.template import BaseTemplate, ReasoningTemplate
from rm_gallery.core.utils.registry import RewardRegistry


class HelpfulnessTemplate(ReasoningTemplate):
    """
    A template class for evaluating the helpfulness of an answer.
    """

    helpfulness: str = Field(default=..., description="is answer helpful? Yes or No.")

    @classmethod
    def format(cls, desc: str, query: str, answer: str) -> str:
        return f"""# Task Description
        {desc}
        # Answer
        {answer}
        # Output Format
        {cls.schema()}
        """


@RewardRegistry.register("helpfulness")
class HelpfulnessReward(LLMModule, PointModule):
    """
    A reward module class for evaluating the helpfulness of an answer using an LLM (Large Language Model).
    """

    name: str = Field(default=...)
    desc: str | None = Field(
        default="Your task is to judge whether the answer is helpfull"
    )
    weight: float = Field(default=1.0, description="weight")
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BaseTemplate] | str | dict = Field(default=HelpfulnessTemplate)

    def _before_call(self, sample: DataSample, **kwargs) -> dict:
        return {
            "desc": self.desc,
            "query": sample.input[-1].content,
            "answer": sample.output[-1].answer.content,
        }

    def _after_call(
        self, response: HelpfulnessTemplate, **kwargs
    ) -> ModuleResult[DimensionScore]:
        return ModuleResult(
            module_name=self.name,
            reward_details=[
                DimensionScore(
                    name="helpfulness",
                    score=1 if response.helpfulness == "Yes" else 0,
                    reason=response.reason,
                    weight=self.weight,
                )
            ],
        )
