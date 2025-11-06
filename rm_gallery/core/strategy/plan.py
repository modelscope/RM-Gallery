from typing import List

from pydantic import BaseModel, Field

from rm_gallery.core.data import DataSample
from rm_gallery.core.grader import Grader, GraderScore
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.template import ChatTemplate, RequiredField, Template
from rm_gallery.core.strategy.base import GraderStrategy


class Plan(BaseModel):
    name: str = Field(..., description="The name of grader to use for evaluation")
    reason: str = Field(..., description="The reason for the plan")


class PlanStrategy(GraderStrategy):
    """Plan strategy that routes to different graders based on classifier.

    This strategy analyzes the input query to determine its intent and selects
    the most appropriate grader for evaluation.
    """

    def __init__(
        self, classifier: ChatTemplate | dict, graders: List[Grader], **kwargs
    ):
        """Initialize PlanStrategy.

        Args:
            classifier (ChatTemplate | dict): The classifier to use for routing.
            graders (List[Grader]): The list of graders to use for evaluation.
        """
        super().__init__(**kwargs)
        self.classifier = (
            classifier
            if isinstance(classifier, ChatTemplate)
            else ChatTemplate(**classifier)
        )
        self.graders = graders

    def __name__(self) -> str:
        """Get the name of the strategy."""
        return "plan"

    async def __call__(
        self, data_sample: DataSample, *args, **kwargs
    ) -> List[GraderScore]:
        """Evaluate a data sample using the selected grader.

        Args:
            data_sample (DataSample): The data sample to evaluate.

        Returns:
            List[GraderScore]: The list of grader scores.
        """
        descriptions = [
            f"{grader.name}: {grader.description}"
            for i, grader in enumerate(self.graders)
        ]
        graders_description = "\n".join(descriptions)
        intent = self.classifier(
            graders_description=graders_description,
            chat_output=Plan,
            **data_sample.data,
        )
        grader = self.graders[intent]
        return await grader(data_sample, *args, **kwargs)


def test_plan_strategy():
    """Test the plan strategy."""
    classifier = ChatTemplate(
        model={
            "model_name": "qwen-plus",
            "stream": False,
            "client_args": {
                "timeout": 60,
            },
        },
        template=Template(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant that helps users to choose the best grader for their evaluation.",
                ),
                ChatMessage(
                    role="user",
                    content="""
Please choose the best grader for the following query: {query}

The graders are:
{graders_description}

Please respond with the name of the grader in the format:
```json
{
    "name": "grader_name"
    "reason": "reason for the choice"
}
```
""",
                ),
            ],
            required_fields=[
                RequiredField(
                    name="query",
                    type="string",
                    position="data",
                    description="The query to evaluate",
                ),
                RequiredField(
                    name="graders_description",
                    type="string",
                    position="others",
                    description="The description of the graders",
                ),
            ],
        ),
    )

    PlanStrategy(
        classifier=classifier,
        graders=[
            Grader(name="grader1", description="grader1 description"),
            Grader(name="grader2", description="grader2 description"),
            Grader(name="grader3", description="grader3 description"),
        ],
    )
