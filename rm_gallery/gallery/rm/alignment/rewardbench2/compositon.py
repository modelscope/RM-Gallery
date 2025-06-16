from pydantic import Field

from rm_gallery.core.reward.composition import RouterComposition


class RewardBench2Composition(RouterComposition):
    name: str = Field(default=..., description="rwardbench2 reward composition")
