"""
LLM as Judge Evaluation Module

Provides LLM-based evaluation for various tasks with multiple scoring modes.

Supported modes:
- Pairwise: winrate, copeland, dgr (TFAS), elo
- Pointwise: direct scoring (1-10)
- Listwise: ranking
"""

from rm_gallery.gallery.evaluation.llm_judge.rewards.alignment import AlignmentReward

__all__ = ["AlignmentReward"]
