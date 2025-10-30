"""
Core reward module exports.

This module provides all reward-related classes and utilities.
"""

# Base reward classes
from rm_gallery.core.reward.base import (
    BaseListWiseReward,
    BaseListWiseRubricReward,
    BaseLLMReward,
    BasePairWiseReward,
    BasePointWiseReward,
    BasePointWiseRubricReward,
    BaseReward,
    BaseRubricReward,
    BaseStepWiseReward,
)

# Reward schemas
from rm_gallery.core.reward.schema import (
    RewardDimensionWithRank,
    RewardDimensionWithScore,
    RewardResult,
)

# VLM reward classes
from rm_gallery.core.reward.vlm_reward import (
    BaseListWiseVLMReward,
    BasePairWiseVLMReward,
    BasePointWiseVLMReward,
    BaseVLMReward,
)

__all__ = [
    # Base rewards
    "BaseReward",
    "BaseStepWiseReward",
    "BasePointWiseReward",
    "BaseListWiseReward",
    "BasePairWiseReward",
    "BaseLLMReward",
    "BaseRubricReward",
    "BasePointWiseRubricReward",
    "BaseListWiseRubricReward",
    # VLM rewards
    "BaseVLMReward",
    "BasePointWiseVLMReward",
    "BaseListWiseVLMReward",
    "BasePairWiseVLMReward",
    # Schemas
    "RewardResult",
    "RewardDimensionWithScore",
    "RewardDimensionWithRank",
]
