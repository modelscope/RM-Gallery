from typing import List

from pydantic import Field

from rm_gallery.core.reward.base import (
    BaseListWisePrincipleReward,
    BasePointWisePrincipleReward,
)
from rm_gallery.core.reward.registry import RewardRegistry

PRINCIPLES = []
SCENARIO = """The assitant aims to provide helpful and accurate responses to the user's questions."""


@RewardRegistry.register("rmb_helpfulness_pointwise")
class HelpfulnessPointWiseReward(BasePointWisePrincipleReward):
    scenario: str = Field(default=SCENARIO)
    principles: List[str] = Field(default=PRINCIPLES)


@RewardRegistry.register("rmb_helpfulness_listwise")
class HelpfulnessListWiseReward(BaseListWisePrincipleReward):
    #     desc: str = Field(default="""Please act as an impartial judge and evaluate the quality of the answers provided by some assistants to the user question displayed below.
    # You need to select exactly 5 appropriate princiles to evaluate the answers based on the current scenario and query.
    # You should critically and accurately assess the assistant’s answer with the key principles to be a qualified response without any potential bias and choose the assistant that follows the user’s query and answers the user’s question best.
    # Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
    # Do not allow the length of the responses to influence your evaluation.
    # Do not favor certain names of the assistants.
    # Be as goal as possible.""")
    scenario: str = Field(default=SCENARIO)
    principles: List[str] = Field(default=PRINCIPLES)
