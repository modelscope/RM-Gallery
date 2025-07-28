import re
from typing import Any, Dict

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BaseLLMReward, BasePointWiseReward
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult
from rm_gallery.core.reward.template import BasePromptTemplate

USER_PROMPT = """Based on the conversation trajectory above, evaluate the task completion quality using the framework provided.

Your evaluation should address:

**Task Understanding**: Does the agent correctly interpret the requirements and objectives?
**Strategic Planning**: Is the approach logical, efficient, and well-structured?
**Execution Quality**: Are the actions appropriate, accurate, and effectively implemented?
**Completion Level**: To what extent are the task goals achieved?
**Error Recovery**: How well does the agent handle mistakes and adapt?

Provide your detailed analysis first, explaining your reasoning for each evaluation dimension. Then assign a precise continuous score between 0.0 and 1.0, where:
- 0.9-1.0: Exceptional performance with complete success
- 0.7-0.9: Strong performance with minor issues
- 0.5-0.7: Adequate performance with notable gaps
- 0.3-0.5: Poor performance with major deficiencies
- 0.1-0.3: Very poor performance with minimal progress
- 0.0-0.1: Complete failure or no meaningful attempt

First provide your detailed reasoning analysis, then output a continuous score between 0.0 and 1.0 enclosed in <reward></reward> tags, e.g., <reward>0.75</reward>
"""


class TrajEvalTemplate(BasePromptTemplate):
    """Template class for trajectory evaluation"""

    @classmethod
    def format(cls, trajectory_text: str, **kwargs) -> str:
        """Format trajectory and evaluation prompt"""
        return f"{trajectory_text}\n\n{USER_PROMPT}"

    @classmethod
    def parse(cls, response: str) -> Dict[str, Any]:
        """Parse LLM response"""
        return {"raw_response": response}


@RewardRegistry.register("trajectory_process")
class TrajProcessReward(BaseLLMReward, BasePointWiseReward):
    """
    Agent trajectory evaluation using LLM as judge for multi-turn conversation quality assessment.

    Evaluates task completion quality across five dimensions: task understanding, strategic planning,
    execution quality, completion level, and error recovery. Returns continuous scores between 0.0-1.0.
    """

    def __init__(self, **kwargs):
        super().__init__(name="trajectory_process", template=TrajEvalTemplate, **kwargs)

    def _pack_message_from_data_sample(self, sample: DataSample) -> str:
        """Pack messages from DataSample

        Args:
            sample: Data sample containing input and output

        Returns:
            str: Formatted trajectory text
        """
        trajectory_text = (
            "The following is the dialogue trace of the task execution:\n\n"
        )

        for msg in sample.input:
            trajectory_text += f"{msg.role.upper()}: {msg.content}\n\n"

        for output in sample.output:
            if output.steps:
                for step in output.steps:
                    trajectory_text += f"{step.role.upper()}: {step.content}\n\n"

            if output.answer:
                if output.answer.content != output.steps[-1].content:
                    trajectory_text += (
                        f"{output.answer.role.upper()}: {output.answer.content}\n\n"
                    )

        return trajectory_text

    def _before_evaluate(self, **kwargs) -> Dict[str, Any]:
        """Prepare evaluation parameters"""
        if "sample" in kwargs:
            trajectory_text = self._pack_message_from_data_sample(kwargs["sample"])
        else:
            raise ValueError("Must provide 'sample' parameter")

        return {"trajectory_text": trajectory_text}

    def _after_evaluate(self, response: Dict[str, Any], **kwargs) -> RewardResult:
        """Process LLM response and extract reward score"""
        raw_response = response.get("raw_response", "")
        reward_pattern = r"<reward>([\d\.]+)</reward>"
        reward_match = re.search(reward_pattern, raw_response.strip())

        if reward_match:
            try:
                score = float(reward_match.group(1))
                score = max(0.0, min(1.0, score))  # Ensure score is within 0-1 range

                reason_text = re.sub(reward_pattern, "", raw_response).strip()

                reward_dimension = RewardDimensionWithScore(
                    name="trajectory_quality",
                    score=score,
                    reason=reason_text
                    if reason_text
                    else "Based on LLM evaluation of task understanding, strategic planning, execution quality, completion level, and error recovery",
                )

                return RewardResult(
                    name=self.name,
                    details=[reward_dimension],
                    extra_data={
                        "raw_response": raw_response,
                        "extracted_score": score,
                        "evaluation_prompt": USER_PROMPT,
                    },
                )
            except ValueError as e:
                logger.error(
                    f"Unable to parse score: {reward_match.group(1)}, error: {e}"
                )
        else:
            logger.warning(f"Unable to extract score from response: {raw_response}")

        # Return default score
        reward_dimension = RewardDimensionWithScore(
            name="trajectory_quality",
            score=0.0,
            reason="Default score due to parsing failure from LLM response",
        )

        return RewardResult(
            name=self.name,
            details=[reward_dimension],
            extra_data={
                "raw_response": raw_response,
                "error": "Failed to parse reward score",
                "default_score": 0.0,
            },
        )
