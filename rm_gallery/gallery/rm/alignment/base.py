from typing import List

from pydantic import Field

from rm_gallery.core.reward.base import BaseListWisePrincipleReward
from rm_gallery.core.reward.registry import RewardRegistry

DEFAULT_HELPFULNESS_DESC = ""
DEFAULT_HELPFULNESS_SCENARIO = ""
DEFAULT_HELPFULNESS_PRINCIPLES = []

DEFAULT_HARMLESSNESS_DESC = ""
DEFAULT_HARMLESSNESS_SCENARIO = ""
DEFAULT_HARMLESSNESS_PRINCIPLES = []

DEFAULT_HONESTY_DESC = ""
DEFAULT_HONESTY_SCENARIO = ""
DEFAULT_HONESTY_PRINCIPLES = []


@RewardRegistry.register("base_helpfulness_listwise")
class BaseHelpfulnessListwiseReward(BaseListWisePrincipleReward):
    desc: str = Field(default=DEFAULT_HELPFULNESS_DESC)
    scenario: str = Field(
        default=DEFAULT_HELPFULNESS_SCENARIO, description="assistant scenario"
    )
    principles: List[str] = Field(default=DEFAULT_HELPFULNESS_PRINCIPLES)


@RewardRegistry.register("base_harmlessness_listwise")
class BaseHarmlessnessListwiseReward(BaseListWisePrincipleReward):
    desc: str = Field(default=DEFAULT_HARMLESSNESS_DESC)
    scenario: str = Field(
        default=DEFAULT_HARMLESSNESS_SCENARIO, description="assistant scenario"
    )
    principles: List[str] = Field(default=DEFAULT_HARMLESSNESS_PRINCIPLES)


@RewardRegistry.register("base_honesty_listwise")
class BaseHonestyListeReward(BaseListWisePrincipleReward):
    desc: str = Field(default=DEFAULT_HONESTY_DESC)
    scenario: str = Field(
        default=DEFAULT_HONESTY_SCENARIO, description="assistant scenario"
    )
    principles: List[str] = Field(default=DEFAULT_HONESTY_PRINCIPLES)


# TODO: add HHH pointwise reward
