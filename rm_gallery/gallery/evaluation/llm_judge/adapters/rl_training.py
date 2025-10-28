"""
RL Training Adapter

Convenience imports for RL training frameworks (VERL, OpenRLHF, etc.)
The actual implementation is framework-agnostic and works with any RL framework.
"""

from rm_gallery.gallery.evaluation.llm_judge.rewards.alignment import AlignmentReward

__all__ = ["AlignmentReward"]
