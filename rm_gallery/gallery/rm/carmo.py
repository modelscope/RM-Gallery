# https://arxiv.org/pdf/2410.21545


from typing import List

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BaseListWiseReward, BaseLLMReward
from rm_gallery.core.reward.schema import RewardDimensionWithRank, RewardResult
from rm_gallery.core.reward.template import BasePromptTemplate, RubricListWiseTemplate


class CriteriaGenerationPrompt(BasePromptTemplate):
    rubrics: List[str] = Field(
        default=...,
        description="""```json
[
    "rubric 1",
    ...
]
```""",
    )

    @classmethod
    def format(
        cls,
        instruction: str,
        **kwargs,
    ) -> str:
        return f"""# Task Description
- You are an impartial judge tasked with generating rubrics for evaluating responses provided by AI
assistants to an instruction.
- Your job is to identify important rubrics, along with detailed descriptions, that a human would use
to objectively evaluate the quality of the response based on the given instruction.
- The rubrics should ensure that responses accurately fulfill the requirements of the instruction.
- The rubrics should be designed to ensure that responses are honest, helpful, and harmless (do not
contain offensive content).
- The descriptions of the rubrics should be framed as chain-of-thought detailed questions that assess
whether the response meets the user’s instruction.
- The length of the response should only be considered a rubric if it is specified in the instruction.

# Input
# Instruction
{instruction}

# Output Requirements
{cls.schema(**kwargs)}
"""


class RelativeEvaluationPrompt(RubricListWiseTemplate):
    best: int = Field(
        default=...,
        description="which completion is the best? just give the number here!!!",
    )

    @classmethod
    def format(cls, instruction, rubrics, completions, **kwargs) -> str:
        completion_str = ""
        for i, completion in enumerate(completions):
            completion_str += f"### Completion {i + 1}\n{completion}\n\n"

        rubrics = "\n".join([f"{i+1}. {rubric}" for i, rubric in enumerate(rubrics)])

        return f"""Task Description
- Please act as an impartial judge and evaluate the quality of the responses provided by two AI
assistants to the user instruction shown below. You should choose the assistant that follows the
user’s instructions and answers the user’s instruction better.
- Your evaluation should consider the provided rubrics.
- Provide detailed reasons assessing the quality of the responses based on each rubric individually.
Clearly specify which assistant performed better for each rubric.
- After assessing all rubrics, provide a final verdict based on the overall performance of the
assistants.
- Don’t be influenced by the order in which the responses are presented. Do not favor certain names
of the assistants. Be as objective as possible.

# Input
## Instruction
{instruction}

## Rubrics
{rubrics}

## Completions
{completion_str}

# Output Requirements
{cls.schema(**kwargs)}
"""


class CARMO(BaseLLMReward, BaseListWiseReward):
    """Context-Aware Reward Modeling"""

    def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
        instruction = sample.input[-1].content

        query = CriteriaGenerationPrompt.format(instruction=instruction)
        response = self.llm.simple_chat(query)
        rubrics = CriteriaGenerationPrompt.parse(response).rubrics
        completions = [output.answer.content for output in sample.output]

        return dict(
            rubrics=rubrics,
            instruction=instruction,
            completions=completions,
        )

    def _after_evaluate(
        self, response: RelativeEvaluationPrompt, sample: DataSample, **kwargs
    ) -> RewardResult:
        """
        Converts LLM response to list-wise ranking metrics.

        Parameters:
            response (RelativeEvaluationPrompt): Parsed LLM comparison

        Returns:
            RewardResult: Relative ranking of responses
        """
        scores = [0 for i in range(len(sample.output))]
        scores[response.best - 1] = 1
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name, reason=response.reason, rank=scores
                )
            ],
        )
