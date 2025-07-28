"""
Tests for agent trajectory evaluation module.

Tests the LlmAsJudgeRewardCalculator class and its main entry methods.
"""

import os

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.schema import RewardResult
from rm_gallery.gallery.rm.agent.trajectory_process import (
    TrajEvalTemplate,
    TrajProcessReward,
)


def create_trajectory_sample() -> DataSample:
    """Create a complex trajectory data sample - includes tool calls"""
    from rm_gallery.core.model.message import ChatTool, Tool

    return DataSample(
        unique_id="complex_traj",
        input=[
            ChatMessage(
                role=MessageRole.USER,
                content="I need to write a Python function to sort a list",
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I can help you write a sorting function",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="It needs to support both ascending and descending order",
            ),
        ],
        output=[
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content="Completed the implementation and testing of the sorting function. The function correctly handles both ascending and descending sorting.",
                ),
                steps=[
                    Step(
                        role=MessageRole.ASSISTANT,
                        content="First, let's analyze the requirements: we need a sorting function that supports both ascending and descending order. I'll use the code editor to create this function",
                        tool_calls=[
                            ChatTool(
                                id="call_editor_001",
                                function=Tool(
                                    name="code_editor",
                                    arguments='{"action": "create", "file": "sort_function.py", "content": "def sort_list(lst, reverse=False):\\n    return sorted(lst, reverse=reverse)"}',
                                ),
                                type="function",
                            )
                        ],
                    ),
                    Step(
                        role=MessageRole.FUNCTION,
                        name="code_editor",
                        content="File sort_function.py created",
                    ),
                    Step(
                        role=MessageRole.ASSISTANT,
                        content="Now let me test this function",
                        tool_calls=[
                            ChatTool(
                                id="call_python_001",
                                function=Tool(
                                    name="python_executor",
                                    arguments='{"code": "from sort_function import sort_list\\n\\n# Test ascending order\\ntest_list = [3, 1, 4, 1, 5]\\nprint(\\"Ascending:\\", sort_list(test_list))\\n\\n# Test descending order\\nprint(\\"Descending:\\", sort_list(test_list, reverse=True))"}',
                                ),
                                type="function",
                            )
                        ],
                    ),
                    Step(
                        role=MessageRole.FUNCTION,
                        name="python_executor",
                        content="Ascending: [1, 1, 3, 4, 5]\nDescending: [5, 4, 3, 1, 1]",
                    ),
                ],
            )
        ],
        task_category="coding",
        source="test",
    )


class TestTrajectoryEvaluationTemplate:
    """Test trajectory evaluation template class"""

    def test_format(self):
        """Test template formatting functionality"""
        template = TrajEvalTemplate()
        trajectory_text = "USER: Hello\nASSISTANT: Hi there!"

        formatted = template.format(trajectory_text=trajectory_text)

        assert trajectory_text in formatted
        assert "Based on the conversation trajectory above" in formatted
        assert "Task Understanding" in formatted
        assert "<reward>" in formatted

    def test_parse(self):
        """Test template parsing functionality"""
        template = TrajEvalTemplate()
        response = "This is a test response."

        parsed = template.parse(response)

        assert isinstance(parsed, dict)
        assert "raw_response" in parsed
        assert parsed["raw_response"] == response


class TestTrajProcessReward:
    def test_init(self):
        reward = TrajProcessReward()

        assert reward.name == "trajectory_process"
        assert reward.template == TrajEvalTemplate
        assert reward.max_retries == 3
        assert reward.llm is None

    def test_after_evaluate_valid_response(self):
        reward = TrajProcessReward()
        response = {
            "raw_response": "This is a detailed analysis process...\n\n<reward>0.85</reward>"
        }

        result = reward._after_evaluate(response=response)

        assert isinstance(result, RewardResult)
        assert result.name == "trajectory_process"
        assert len(result.details) == 1
        assert result.details[0].name == "trajectory_quality"
        assert result.details[0].score == 0.85
        assert "extracted_score" in result.extra_data
        assert result.extra_data["extracted_score"] == 0.85

    def test_after_evaluate_invalid_score(self):
        """Test invalid score handling"""
        reward = TrajProcessReward()
        response = {"raw_response": "<reward>invalid</reward>"}

        result = reward._after_evaluate(response=response)

        assert result.details[0].score == 0.0
        assert "error" in result.extra_data
        assert result.extra_data["default_score"] == 0.0

    def test_after_evaluate_no_reward_tag(self):
        """Test response without reward tag"""
        reward = TrajProcessReward()
        response = {"raw_response": "This is a response without a score tag"}

        result = reward._after_evaluate(response=response)

        assert result.details[0].score == 0.0
        assert "error" in result.extra_data

    def test_evaluate_trajectory(self):
        """Integration test: complex trajectory evaluation"""
        llm = OpenaiLLM(
            model="qwen3-32b",
            enable_thinking=False,
            base_url="http://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        reward = TrajProcessReward(llm=llm)
        sample = create_trajectory_sample()

        result = reward.evaluate(sample)

        assert isinstance(result, DataSample)
        assert len(result.output[0].answer.reward.details) == 1
        assert 0.0 <= result.output[0].answer.reward.details[0].score <= 1.0
        assert result.output[0].answer.reward.details[0].reason is not None
        assert len(result.output[0].answer.reward.details[0].reason) > 0
