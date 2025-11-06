# https://arxiv.org/pdf/2410.21545


from typing import Any, Dict, List

from rm_gallery.core.data import DataSample
from rm_gallery.core.grader import GraderMode, GraderScore, LLMGrader
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.template import ChatTemplate, RequiredField, Template
from rm_gallery.core.model.utils import _json_loads_with_repair

CriteriaGenerationTemplate = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are an impartial judge tasked with generating rubrics for evaluating responses provided by AI assistants to an instruction.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
- Your job is to identify important rubrics, along with detailed descriptions, that a human would use to objectively evaluate the quality of the response based on the given instruction.
- The rubrics should ensure that responses accurately fulfill the requirements of the instruction.
- The rubrics should be designed to ensure that responses are honest, helpful, and harmless (do not contain offensive content).
- The descriptions of the rubrics should be framed as chain-of-thought detailed questions that assess whether the response meets the user's instruction.
- The length of the response should only be considered a rubric if it is specified in the instruction.\n
# Input
## Instruction
{instruction}

# Output Requirements
```json
{
    "rubrics": [
        "rubric 1",
        ...
    ]
}
```
""",
        ),
    ],
    required_fields=[
        RequiredField(
            name="instruction",
            type="string",
            position="data",
            description="The instruction to evaluate responses for",
        )
    ],
)


RelativeEvaluationTemplate = Template(
    messages=[
        ChatMessage(
            role="system",
            content="Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user instruction shown below.",
        ),
        ChatMessage(
            role="user",
            content="""Task Description
- Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user instruction shown below. You should choose the assistant that follows the user's instructions and answers the user's instruction better.
- Your evaluation should consider the provided rubrics.
- Provide detailed reasons assessing the quality of the responses based on each rubric individually. Clearly specify which assistant performed better for each rubric.
- After assessing all rubrics, provide a final verdict based on the overall performance of the assistants.
- Don't be influenced by the order in which the responses are presented. Do not favor certain names of the assistants. Be as objective as possible.\n
# Input
## Instruction
{instruction}

## Rubrics
{rubrics}

## Completion
{completion}

# Output Requirements
```json
{
    "rank": "The rank of each completions",
    "reason": "The reason for the rank."
}
```
""",
        ),
    ],
    required_fields=[
        RequiredField(
            name="instruction",
            type="string",
            position="data",
            description="The instruction to evaluate responses for",
        ),
        RequiredField(
            name="rubrics",
            type="string",
            position="grader",
            description="The rubrics for evaluation",
        ),
        RequiredField(
            name="completion",
            type="string",
            position="sample",
            description="The completion to evaluate",
        ),
    ],
)


class Cramo(LLMGrader):
    def __init__(
        self,
        name: str = "",
        grader_mode: GraderMode = GraderMode.LISTWISE,
        template: Template | None = RelativeEvaluationTemplate,
        model: Dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(name, grader_mode, template, model, **kwargs)

    async def __call__(
        self, data_sample: DataSample, *args, **kwargs
    ) -> List[GraderScore]:
        chat = ChatTemplate(
            template=CriteriaGenerationTemplate,
            model=self.model,
        )
        rubrics = chat(
            char_output=_json_loads_with_repair, **data_sample.data
        ).metadata.get("rubrics", [])
        return await super().__call__(
            data_sample, rubrics="\n".join(rubrics), *args, **kwargs
        )
