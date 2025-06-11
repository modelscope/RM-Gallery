from typing import List

from pydantic import Field

from rm_gallery.core.reward.base import (
    BaseListWisePrincipleReward,
    BasePointWisePrincipleReward,
)
from rm_gallery.core.reward.registry import RewardRegistry

REWRITE_PRINCIPLES = [
    "Creative Writing: Involves the generation of original content.",
    "Professional Content Generation: Involves the generation of professional content.",
]


@RewardRegistry.register("rewrite_pointwise")
class RewritePointWiseReward(BasePointWisePrincipleReward):
    principles: List[str] = Field(default=REWRITE_PRINCIPLES)


@RewardRegistry.register("rewrite_listwise")
class RewriteListWiseReward(BaseListWisePrincipleReward):
    principles: List[str] = Field(default=REWRITE_PRINCIPLES)
